# 台車型倒立振子: PINO で予測し、MPC で倒立を保つ

## この例題で行うこと

添付図の台車型倒立振子について、**現在の状態と将来の力の時間波形から、短時間先までの運動を予測する作用素**を学習する。その予測器を MPC（モデル予測制御）に組み込み、直立近傍から倒立状態を保つ。

```text
現在の状態を測る
    ↓
将来の力の候補 F(τ) → 学習済み PINO → 将来の状態軌道
    ↑                                  ↓
    └──────── 制御コストを下げるように力を最適化
                                        ↓
                              最初の区間だけ実行
                                        ↓
                             新しい状態を測って繰り返す
```

PINO が担当するのは「力を与えると、どう動くか」の予測。MPC が担当するのは「どの力を与えるか」の決定。PINO の重みは制御中に更新しない。

この系は常微分方程式（ODE）である。PINO 原論文の PDE ベンチマークをそのまま再現するものではなく、**関数入力から軌道への作用素学習と物理損失を、小さな力学系で確かめる教育用の例**とする。

## 1. 図に対応する物理モデル

質量のない剛体棒の先端に質点を置き、台車は水平に動く。摩擦、車輪の回転慣性、棒の質量、空気抵抗は初期モデルでは省略する。

| 記号 | 意味 | 初期値 |
|---|---|---:|
| $M$ | 台車の質量 | 1.0 kg |
| $m$ | 先端質点の質量 | 0.1 kg |
| $l$ | 支点から質点までの長さ | 0.5 m |
| $g$ | 重力加速度 | 9.81 m/s² |
| $x$ | 台車の水平位置。右が正 | 目標 0 m |
| $\theta$ | 鉛直上向きからの角度。左傾きが正 | 目標 0 rad |
| $F$ | 台車に加える水平力。右が正 | 上限 ±10 N |

図の左に傾いた姿勢を正の角度で表す。質点の座標は

$$
X=x-l\sin\theta,\qquad Y=l\cos\theta.
$$

$l$ は棒の全長であり、均一な棒の重心までの距離ではない。

状態の順序は全ファイルで

$$
s=[x,v,\theta,\omega]^\mathsf T,
\qquad v=\dot x,\quad\omega=\dot\theta
$$

とする。非線形運動方程式は

$$
(M+m)\ddot x-ml\cos\theta\,\ddot\theta
+ml\sin\theta\,\dot\theta^2=F,
$$

$$
l\ddot\theta-\cos\theta\,\ddot x-g\sin\theta=0.
$$

したがって、加速度を明示すると

$$
a_x(s,F)=\frac{F+mg\sin\theta\cos\theta
-ml\sin\theta\,\omega^2}{M+m\sin^2\theta},
$$

$$
a_\theta(s,F)=\frac{\cos\theta\,a_x(s,F)+g\sin\theta}{l}.
$$

$$
\dot s=f(s,F)=[v,a_x,\omega,a_\theta]^\mathsf T.
$$

直立のまま右向きに力を加えると、支点が右へ動いて質点が相対的に左へ傾くため、$\ddot\theta>0$ となる。符号の確認点として使える。

既存資料の式を引用する際は角度の原点に注意する。参考文献の MIT ノートは下向きが角度ゼロであり、本例の左正・上向きゼロへの変換は $\theta_{\mathrm{MIT}}=\pi+\theta$ である。

## 2. PINO が学習する作用素

制御時刻 $t$ からの相対時間を $\tau\in[0,H]$ とし、

$$
\mathcal G_\phi:(s_t,F(\cdot))\longmapsto
\hat s(\cdot),\qquad H=0.4\ \mathrm{s}
$$

を学ぶ。入力する力は、9 個の節点の間を線形補間した時間関数である。物理パラメータは固定し、現在の状態と力の波形を変えて学習する。

- 入力チャンネル: $s_t$ の4成分、$F(\tau)$、$\tau/H$ の計6成分。
- バックボーン: 時間軸に沿う1次元 FNO。幅32、12モード、3層。
- 生の出力: 加速度 $[\hat a_x(\tau),\hat a_\theta(\tau)]$。
- 最終出力: 加速度を2回積分して得る $[\hat x,\hat v,\hat\theta,\hat\omega]$。

具体的には

$$
\hat v(\tau)=v_t+\int_0^\tau\hat a_x(\xi)d\xi,
\qquad
\hat x(\tau)=x_t+\int_0^\tau\hat v(\xi)d\xi,
$$

$$
\hat\omega(\tau)=\omega_t+\int_0^\tau\hat a_\theta(\xi)d\xi,
\qquad
\hat\theta(\tau)=\theta_t+\int_0^\tau\hat\omega(\xi)d\xi.
$$

実装は微分可能な累積台形則である。初期状態は構造で一致し、離散的な運動学の積分関係も満たす。連続時間の方程式を厳密に満たす保証はなく、積分の離散化誤差が残る。FNO の非周期時間端の影響を抑えるため、各層を通す前に25%のゼロパディングを入れる。

将来の力を全区間入力するので、このネットワーク自体は時間についての因果マスクを持たない。MPC では最適化中の力の候補を全区間知っているため、その候補に対する予測器として使用できる。未知の将来外力を入力したことにはならない。

## 3. 物理損失と教師データ

加速度の物理残差を

$$
r_x=\frac{\hat a_x-a_x(\hat s,F)}{g},\qquad
r_\theta=\frac{\hat a_\theta-a_\theta(\hat s,F)}{g/l}
$$

とし、状態のスケールを

$$
D_s=\operatorname{diag}(l,\sqrt{gl},1,\sqrt{g/l})
$$

として

$$
\mathcal L_{\mathrm{data}}
=\mathbb E\left[\|D_s^{-1}(\hat s-s^{\mathrm{RK4}})\|^2\right],
\qquad
\mathcal L_{\mathrm{phys}}
=\mathbb E[r_x^2+r_\theta^2],
$$

$$
\mathcal L=\mathcal L_{\mathrm{data}}
+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}
$$

を用いる。実装の平均はそれぞれ状態4成分、残差2成分についても取る。デフォルトは $\lambda_{\mathrm{phys}}=1$。初期条件の罰則は積分構造で満たすので設けない。

教師データは非線形 ODE の RK4 解。力は各区間で線形に変化させ、RK4 の中間時刻でもその力を使う。学習は65点の予測と物理残差に対し、教師の損失だけを4点おきの17点で計算する。

| 項目 | 設定 |
|---|---|
| 学習初期状態 | $x\in[-0.4,0.4]$ m、$v\in[-1,1]$ m/s、$\theta\in[-0.3,0.3]$ rad、$\omega\in[-1.5,1.5]$ rad/s |
| 力の節点 | 9個、それぞれ一様分布 $[-10,10]$ N |
| 学習 | 独立な初期状態・入力波形512件 |
| 検証 | 別の乱数系列64件。保存する重みを選ぶために使う |
| テスト | さらに別の乱数系列64件。重みの選択後にだけ使う |
| 解像度確認 | 同じテスト入力関数を129点で再評価 |

129点で動作することと、その解像度で精度が保たれることは分けて確認する。`training.json` に両方の誤差を保存する。

## 4. PINO を使う MPC

各制御時刻で9個の力の節点 $u_0,\ldots,u_8$ を変数とし、予測軌道に対する

$$
J=\int_0^H
\left(\hat s(\tau)^\mathsf TQ\hat s(\tau)+R F(\tau)^2\right)d\tau
+\hat s(H)^\mathsf T P\hat s(H)
$$

を小さくする。$Q=\operatorname{diag}(4,1,80,2)$、$R=0.1$ とする。$P$ は同じ $Q,R$ を用いた直立まわりの連続時間 LQR の Riccati 解。

この終端コストは既知の局所線形モデルの情報を使う。したがって本例は、学習した予測器と古典制御を組み合わせた構成である。LQR の制御入力への切り替えや、真の ODE を使った MPC 内部の予測は行わない。

- $u_i=10\tanh z_i$ として入力の上限を守る。
- 凍結した PINO と積分層を通して $J$ を $z_i$ で微分し、L-BFGS で更新する。
- 最初はゼロ入力、2回目以降は前の最適化結果を1節点ずらして初期化する。
- 求めた力の最初の区間 $[0,0.05]$ 秒だけ、線形に変化する力として実際の非線形シミュレータに加える。
- 0.05秒ごとに現在状態を測り直して最適化する。

制御実験では、PINO の予測結果を実際の状態として扱わず、別の RK4 シミュレータで次の状態を計算する。状態4成分を誤差なく測れる設定である。

## 5. 最初の制御課題と判定

初期状態は $x=0.2$ m、$\theta=10$ 度、速度・角速度はゼロ。5秒間制御し、1.5〜1.6秒の間に未知の右向き外力2 Nを追加する。外力は入力上限とは別に台車に加わる。

比較対象は無制御、力を±10 Nに飽和した LQR、PINO + MPC。同じ初期状態・外力を用い、次を別々に記録する。

1. `balanced_entire_run`: 全区間で $|\theta|\le30$ 度、$|x|\le2.4$ mを守ったか。
2. `settled_at_end`: 上記に加え、最後に $|x|<0.1$ m、$|v|<0.1$ m/s、$|\theta|<2$ 度、$|\omega|<0.1$ rad/sを満たすか。
3. 最大角度、最大台車変位、最大操作力、終端状態、計画にかかった時間。
4. 学習した予測と実シミュレータの次の状態との差。外力が加わる区間では外乱の効果も含む。

角度・位置の上限は失敗判定であり、MPC の最適化に状態制約としては入れていない。失敗時はその時点でロールアウトを打ち切り、理由と実行時間を残す。打ち切りが違う試行の累積コストだけを比較しない。

これは直立近傍の安定化であり、下向きからの振り上げは扱わない。有限の試行で成功しても安定性や未知の初期状態での成功を証明したことにはならない。短時間予測の精度、閉ループの成功、LQRに対する優位性、実時間動作は別の評価である。

## 6. 実行方法

プロジェクトルート `/Users/owada9101/code/PIML` で実行する。

```bash
uv sync
uv run python -m unittest Example.cartpole_pino.test_example -v
uv run python -m Example.cartpole_pino.train --steps 1500
uv run python -m Example.cartpole_pino.control
```

すでに依存関係が入った Python 環境なら、`uv run python` を `python3` に置き換えて実行できる。

出力:

```text
Example/cartpole_pino/outputs/
  pino/
    model.pt           学習済みの予測器
    training.json      設定、損失の推移、検証で選んだステップ、テスト誤差
    prediction.npz     テスト入力・正解・予測
    prediction.png     未学習の入力波形1件に対する予測
  control/
    control.json       制御条件、判定、時間、終端状態
    uncontrolled.npz   無制御の軌道（失敗判定時に終了）
    lqr.npz            LQR の軌道
    pino_mpc.npz       PINO + MPC の軌道、次の状態の予測誤差
    control.png        3方式の比較
    control.gif        PINO + MPC のアニメーション
```

データのみの FNO と比較するには、同じシード・ステップ数・モデル構造で物理損失の重みだけを0にする。

```bash
uv run python -m Example.cartpole_pino.train --steps 1500 --physics-weight 0 --out Example/cartpole_pino/outputs/fno
uv run python -m Example.cartpole_pino.control --checkpoint Example/cartpole_pino/outputs/fno/model.pt --out Example/cartpole_pino/outputs/fno_control
```

最初の確認は短い学習でも可能だが、動作確認と精度・制御性能の確認は区別する。

```bash
uv run python -m Example.cartpole_pino.train --steps 10 --train-samples 16 --out Example/cartpole_pino/outputs/smoke
```

`control.py` の `--angle-deg`、`--cart-position`、`--disturbance` で条件を変えられる。デフォルト出力先は同じなので、実験を保存したい場合は `--out` を変える。

## 7. ファイルの役割

| ファイル | 役割 |
|---|---|
| `dynamics.py` | 物理定数、非線形 ODE、RK4、エネルギー、LQR、データ生成 |
| `model.py` | FNO、微分可能な積分、物理残差 |
| `train.py` | 学習、検証での重み選択、未使用テスト入力での評価 |
| `control.py` | 学習済み PINO による MPC、LQR・無制御との比較、可視化 |
| `test_example.py` | 物理モデルの整合性と制御に必要な勾配のチェック |
| `RESULTS.md` | この環境で実行した結果と確認範囲 |

## 8. 次に調べること

- 同じ条件で FNO と PINO を比較し、物理損失が予測誤差・制御成功にどう影響するか。
- 教師データの件数・時間間隔を変えた場合の差。
- 複数シードと複数の初期状態・外力での成功率。
- 学習範囲外の角度や速度での失敗、予測誤差を最適化が利用してしまう現象。
- 計画時間と50 msの制御周期の関係。現状の実装をそのまま実時間制御可能とはみなさない。
- 摩擦やパラメータ差、観測誤差を加えた場合の差。

## 参考

- Li et al., [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794): 作用素学習にデータと物理残差を組み合わせる出発点。
- Tedrake, [Underactuated Robotics, Cart-Pole](https://underactuated.csail.mit.edu/acrobot.html): 力学と倒立制御の参照。角度の原点は本例と異なる。
- `../../Material/physics-informed-ml-note.pdf` の3.8節・3.10節: 関数から作用素への拡張と、条件について平均した物理・データ損失の説明。本例の数値条件・制御実装はこのフォルダで新しく定めたもの。
