# Cavity 流れ (Lid-Driven Cavity Flow)

---

## 1. Cavity 流れとは？

### 概要

Cavity 流れ（蓋駆動空洞流れ）は、正方形または矩形の閉じた容器（キャビティ）の上蓋を一定速度で水平方向にスライドさせることで容器内の流体を駆動する、計算流体力学（CFD）において最も古典的かつ標準的なベンチマーク問題の一つです。

容器の3辺は静止壁（no-slip）であり、上蓋だけが速度 $U = 1.0$ で移動します。この非対称な境界条件が容器内部に渦（翼渦・隅渦）を生成します。

### 物理的特性

流体運動はレイノルズ数 $Re$ によって特徴付けられます。

$$Re = \frac{U L}{\nu}$$

ここで $U$ は蓋速度、$L$ はキャビティ辺長、$\nu$ は動粘性係数（$\nu = 1 / Re$）です。

| $Re$   | 流れの特徴                                     |
|--------|------------------------------------------------|
| 低 $Re$ ($\leq 100$) | 中央に1つの安定した一次渦（Primary Vortex）が形成される。定常解に収束しやすい。 |
| 中 $Re$ ($\approx 1000$) | 下隅に二次渦（Secondary Vortex）が発達し始める。|
| 高 $Re$ ($\geq 3200$) | 定常から非定常（振動的な流れ）へ遷移する。     |

### 検証規範としての位置付け

Cavity 流れは解析解が存在しないものの、Ghia et al. (1982) による高精度数値解が参照標準として広く利用されており、新しいCFDソルバーの検証（Validation & Verification）に不可欠な問題とされています。

本実装の物理設定は以下のとおりです。

| パラメータ          | 値                |
|---------------------|-------------------|
| ドメインサイズ      | $1.0 \times 1.0$  |
| レイノルズ数 $Re$   | 100               |
| 動粘性係数 $\nu$    | $0.01$            |
| 蓋速度 $U$          | $1.0$             |
| グリッド数          | $300 \times 300$  |
| 時間刻み $\Delta t$ | $2.5 \times 10^{-4}$ |

---

## 2. 実装した処理の説明

### 2.1 モジュール構成

```
cavity_flow/
  __init__.py        パッケージエントリポイント。CavityFlowSolver と SolverConfig をエクスポート。
  solver.py          MAC 法によるメインソルバー。
  poisson.py         共役勾配法（CG）によるポアソン方程式・ヘルムホルツ方程式の求解。
  boundary.py        境界条件の適用（no-slip 壁・移動蓋）。
  visualize.py       流線図・圧力コンターの生成。
run_cavity.py        シミュレーションの実行エントリポイント。
tests/
  test_cavity_flow.py  Pytest による単体テスト（境界条件・ソルバー・収束特性）。
```

---

### 2.2 MAC スタガードグリッド

空間離散化には **MAC（Marker-And-Cell）スタガードグリッド** を採用しています。変数を異なる位置に配置することで、速度の発散を厳密に離散化し、チェッカーボード不安定性を回避します。

```
  +-------v[i,j+1]-------+
  |                       |
u[i,j]   p[i,j]        u[i+1,j]
  |                       |
  +-------v[i,j]---------+
```

| 変数           | 配置位置         | 形状            |
|----------------|------------------|-----------------|
| $p_{i,j}$      | セル中心         | $(N_x, N_y)$    |
| $u_{i,j}$      | 垂直セル面（左右）  | $(N_x+1, N_y)$  |
| $v_{i,j}$      | 水平セル面（上下）  | $(N_x, N_y+1)$  |

---

### 2.3 支配方程式

非圧縮性 Navier-Stokes 方程式（無次元化）を解きます。

**運動量方程式:**

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u}$$

**連続方程式（非圧縮性条件）:**

$$\nabla \cdot \mathbf{u} = 0$$

---

### 2.4 タイムステップのアルゴリズム（Operator Splitting）

1タイムステップ $n \to n+1$ は以下のオペレータ分離で処理されます。

#### Step 1: 移流項の陽的更新（Explicit Advection）

移流のみを時間前進します（粘性項はこの段階では除外）。

$$u_{\text{adv}} = u^n + \Delta t \left[ -\left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) \right]$$

$$v_{\text{adv}} = v^n + \Delta t \left[ -\left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) \right]$$

移流項の空間差分には **上流差分法（Upwind Differencing）** を使用し、$u > 0$ なら前進差分、$u < 0$ なら後退差分を選択することで安定性を確保しています。

#### Step 2: 境界条件の適用

移流後の速度場に no-slip 壁（$u = v = 0$）および移動蓋（$u = 1.0$）の境界条件を適用します（`boundary.py`）。

#### Step 3: 陰的拡散処理（Implicit Diffusion via Helmholtz Solve）

粘性項を陰的に処理することで、明示的拡散スキームの安定性制約

$$\Delta t \leq \frac{1}{2\nu \left(1/\Delta x^2 + 1/\Delta y^2\right)}$$

を撤廃し、より大きな $\Delta t$ を許容します。以下のヘルムホルツ方程式を $u^*$、$v^*$ についてそれぞれ解きます。

$$\left(I - \nu \Delta t \, \nabla^2\right) u^* = u_{\text{adv}}$$

$$\left(I - \nu \Delta t \, \nabla^2\right) v^* = v_{\text{adv}}$$

ディリクレ境界条件（ゼロ値）の下でヘルムホルツ方程式は **正定値対称（SPD）** となり、共役勾配法（CG）が無条件収束します（`poisson.py` — `solve_helmholtz_cg`）。

#### Step 4: 境界条件の再適用

陰的拡散後の速度場 $u^*$, $v^*$ に再度境界条件を適用します。

#### Step 5: 圧力ポアソン方程式の求解（Pressure Poisson Solve）

速度場の発散を消去するための圧力補正 $\phi$ を求めます。

**発散の計算:**

$$\nabla \cdot \mathbf{u}^* = \frac{\partial u^*}{\partial x} + \frac{\partial v^*}{\partial y}$$

**圧力ポアソン方程式:**

$$\nabla^2 \phi = \frac{1}{\Delta t} \nabla \cdot \mathbf{u}^*$$

壁面でのノイマン境界条件（ゼロ勾配 $\partial \phi / \partial n = 0$）を適用します。求解には以下の定式化に基づく CG 法を使用します。

$$(-\nabla^2) \phi = -\frac{1}{\Delta t} \nabla \cdot \mathbf{u}^*$$

演算子 $-\nabla^2$（負のラプラシアン）が半正定値となるよう再定式化し、右辺と反復解から定数成分を除去することで零空間（定数関数）を射影処理し、平均ゼロの圧力場を得ます（`poisson.py` — `solve_poisson_cg`）。

#### Step 6: 速度の補正（Velocity Correction）

圧力勾配を用いて内部セル面の速度を補正し、発散ゼロ条件を満たします。

$$u^{n+1}_{i,j} = u^*_{i,j} - \Delta t \frac{\partial \phi}{\partial x} \bigg|_{i,j}$$

$$v^{n+1}_{i,j} = v^*_{i,j} - \Delta t \frac{\partial \phi}{\partial y} \bigg|_{i,j}$$

壁面・蓋面のセル面は境界値を保持します（内部面のみ補正対象）。

#### Step 7: 最終境界条件の適用

速度補正後、全境界面に対して厳密に境界条件を適用します。

#### Step 8: 定常収束の判定

$L^\infty$ ノルムによる速度変化を収束判定に使用します。

$$\delta = \max\left(\|u^{n+1} - u^n\|_\infty,\ \|v^{n+1} - v^n\|_\infty\right)$$

$\delta < \varepsilon_{\text{tol}}$（デフォルト $10^{-6}$）であれば定常解に収束したと判断してシミュレーションを終了します。

---

### 2.5 線形ソルバー（共役勾配法）

ポアソン方程式・ヘルムホルツ方程式のいずれも **共役勾配法（Conjugate Gradient）** で求解します（`poisson.py`）。

| ソルバー関数           | 対象方程式                              | 境界条件 | 備考                          |
|------------------------|----------------------------------------|----------|-------------------------------|
| `solve_poisson_cg`     | $\nabla^2 \phi = r$                    | ノイマン  | 零空間射影でゼロ平均を保証      |
| `solve_helmholtz_cg`   | $(I - c\nabla^2) x = b$                | ディリクレ | $c = \nu \Delta t$、SPD 保証  |

両ソルバーとも PyTorch テンソル演算で実装されており、Apple Silicon の **Metal Performance Shaders（MPS）** デバイスで実行することで GPU 並列演算の恩恵を受けます（`device="mps"` 対応）。

---

### 2.6 境界条件の実装（`boundary.py`）

`apply_boundary_conditions` 関数が以下の条件を適用します。

| 境界           | $u$ 速度      | $v$ 速度      |
|----------------|---------------|---------------|
| **上蓋（移動）** | $u = 1.0$   | $v = 0$       |
| **下壁（固定）** | $u = 0$     | $v = 0$       |
| **左壁（固定）** | $u = 0$     | $v = 0$       |
| **右壁（固定）** | $u = 0$     | $v = 0$       |

---

### 2.7 可視化（`visualize.py`）

`plot_streamlines` 関数がシミュレーション結果を可視化します。

1. スタガードグリッド上の $u$, $v$ をセル中心へ線形補間
2. Matplotlib の `streamplot` で流線図（速度の大きさで色付け）を描画
3. 圧力場コンター（等圧線）をオーバーレイ
4. 指定パスへ PNG 形式で保存

---

### 2.8 実行フロー（`run_cavity.py`）

```python
config = SolverConfig(nx=300, ny=300, re=100.0, dt=2.5e-4, max_steps=10000)
solver = CavityFlowSolver(config, device="mps")  # Apple Silicon MPS
result = solver.run()
plot_streamlines(result["u"], result["v"], result["p"], ...)
```

---

### 参考文献

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
- Harlow, F. H., & Welch, J. E. (1965). Numerical calculation of time-dependent viscous incompressible flow of fluid with a free surface. *Physics of Fluids*, 8(12), 2182–2189.
