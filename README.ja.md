# RamGPT: RamTorchでVRAM 8GB級GPUに合わせたGPT-2 1B訓練テンプレート

## 1. プロジェクト概要
- RamTorchのCPU→GPU非同期ストリーミング線形層をGPT-2ブロックに組み込み、RTX 3050のようなVRAM 8GBクラスでも巨大モデルを回せるようにしました。citeturn0search0
- RamTorch 0.2.1の線形層はCUDA転送ストリームやダブルバッファで重みをGPUにバウンスさせる実装なので、CPUメモリを事実上のメイン・ストアにできます。citeturn0search3
- トレーニング/推論両方でPyTorchのAMPと勾配チェックポイントを利用し、W&Bロギング・Hugging Face Hubへのアップロードフックも搭載しています。
- `--micro-test`フラグまたは`configs/micro_test.yaml`で超軽量設定に切り替え、OOM確認だけ先に走らせられます。

## 2. ディレクトリ構成
```
ramgpt/
├── .venv/                 # プロジェクト専用仮想環境（作成後）
├── configs/
│   ├── train_1b.yaml      # RTX3050 LP 6GB向け設定（約1B）
│   ├── train_0p1b.yaml    # 0.1B級の軽量モデル設定
│   └── micro_test.yaml    # CPUでも動く最小構成
├── requirements.txt       # 主要Python依存
├── src/ramgpt/
│   ├── config.py          # dataclassベースの設定ローダ
│   ├── data.py            # HF Datasetsストリーミング+パッキング
│   ├── model.py           # RamTorch線形層を使うGPT-2本体
│   ├── train.py           # 1B訓練ループ＋W&B/HF連携
│   └── inference.py       # 生成用CLI
└── README.ja.md
```

## 3. セットアップ手順
1. **仮想環境作成**
   ```bash
   cd /path/to/ramgpt
   python3 -m venv .venv --without-pip
   . .venv/bin/activate
   python -m ensurepip --upgrade  # ensurepipがない場合はget-pip.pyを使う
   ```
2. **依存インストール**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   # CUDA 12.4 + RTX3050の場合のTorch例
   pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```
3. **追加ツール**（任意）
   - Weights & Biases: `export WANDB_API_KEY=...`
   - Hugging Face: `export HF_TOKEN=...`

## 4. 設定ファイルのカスタマイズ
- `configs/train_1b.yaml`
  - `data`: 既定で`hotchpotch/fineweb-2-edu-japanese`の`sample_10BT`サブセットをストリーミングします。`max_train_samples`/`max_eval_samples`で1パスに流すサンプル数を制限し、VRAM 6GBでも学習を継続できるようにしています。ローカルTXTを使いたい場合は`dataset_path`を指定し、`dataset_name`を`null`にしてください。
  - `model`: 36層/24ヘッド/1536 hiddenで約1Bパラメータ。`max_position_embeddings`は6GBモデルで現実的な1024トークンに調整しています。
  - `optim` & `scheduler`: マイクロバッチ1×勾配蓄積192で実効バッチを確保しつつ、LRは2e-4からコサイン減衰します。
  - `checkpoint`: `hf_repo_id`を設定すると自動でHubにpushします（`HF_TOKEN`または`checkpoint.hf_token`が必要）。
  - `logging`: `use_wandb: true`なら `WANDB_API_KEY` を用意してください。オフライン確認だけなら `WANDB_MODE=offline` を推奨。
- `configs/train_0p1b.yaml`
  - `data`: 1,000,000件程度を1パスに使う想定で`max_train_samples`を小さめに設定しています。`seq_len=768`＆マイクロバッチ2で、RTX3050でもCPUメモリ使用量が10GB前後に収まります。
  - `model`: 18層/12ヘッド/768 hiddenで約0.11Bパラメータ。`ram_cpu_buffer`も2バッファに削ってCPU↔GPU転送を軽量化しています。
  - `optim`: 基本LR 4e-4、勾配蓄積64で実効バッチ約98kトークン。
  - `bf16: false / fp16: true`でAmpere世代のFP16 Tensor Coreを直接利用します（RTX30はBF16サポートが限定的なため）。

## 5. トレーニングの実行
### 5.1 OOMスモークテスト（推奨）
```bash
. .venv/bin/activate
python -m ramgpt.train --config configs/micro_test.yaml --micro-test
```
- 2層・シーケンス長128でCPUでも数分以内に終了します。
- CUDAが無い環境ではRamTorch層は自動的に通常の`nn.Linear`にフォールバックします。

### 5.2 実トレーニング（FineWeb2 Edu JPストリーミング）
```bash
. .venv/bin/activate
HF_DATASETS_CACHE=/mnt/ssd/hf_cache \
python -m ramgpt.train --config configs/train_1b.yaml
```
- RTX3050 LP 6GBを想定し、**マイクロバッチ1 × 勾配蓄積192**で実効バッチを作ります。
- `sample_10BT`サブセットは約100億トークンなので、`max_train_samples`を変えてパス長を調整してください。
- `huggingface-cli login`済みだとHFストリーミングのリトライが安定します。
- 進捗は`tqdm`とW&B両方に表示。W&Bを切りたい場合はコンフィグで`logging.use_wandb: false`。

### 5.3 RTX 3050 LP 6GB向けメモ
- GPU VRAM 6GBでは勾配と活性化をほぼすべてCPUオフロードする必要があるため、`gradient_checkpointing`と`bf16`（ハードが対応していれば）を維持してください。
- 1ステップの実効トークン数は `seq_len (1024) × micro_batch_size (1) × grad_accum (192) = 196,608`。この値を増減したい場合は`grad_accum`を優先的に変えると安定します。
- さらに余裕が欲しい場合は`data.seq_len=768`や`model.n_layer=32`に落としても動作します（RAM／学習速度と相談）。

### 5.4 チェックポイント管理
- `checkpoints/step_0000100.pt`のようなファイルが生成され、`tokenizer/`配下にトークナイザ設定を保存。
- `checkpoint.keep_last_n`で古いファイルを自動削除。
- `push_to_hub_interval`毎に`checkpoint.hf_repo_id`へ`HF_TOKEN`を使ってアップロードします。

## 6. 推論
```bash
. .venv/bin/activate
python -m ramgpt.inference \
  --checkpoint checkpoints/step_0001000.pt \
  --prompt "ヘルスケア分野におけるLLMの活用について教えて" \
  --max-new-tokens 200 --temperature 0.8 --top-k 40
```
- `--tokenizer`で独自トークナイザを指定可能。未指定なら`checkpoint/tokenizer`→HF `gpt2`の順で探索します。
- 出力全文が標準出力に流れるので、必要に応じて`tee`で保存してください。

## 7. Weights & Biases ログ
- `logging.use_wandb: true`の際に自動で`wandb.init()`されます。
- 手元のRTX3050からダッシュボードを確認するだけなら`WANDB_MODE=offline`でローカルにログ→後で`wandb sync`が可能。
- `wandb_project`や`wandb_run_name`はYAMLから変更できます。

## 8. Hugging Face Hub 連携
1. `checkpoint.hf_repo_id: <username>/<repo>`を設定。
2. `HF_TOKEN`環境変数、または`checkpoint.hf_token`にPATを指定。
3. 設定済みであれば`push_to_hub_interval`ステップ毎に`checkpoints/`フォルダ全体が自動アップロードされます（`.pt`や`.yaml`、`tokenizer/`が対象）。

## 9. データセットメモ（FineWeb2 Edu Japanese）
- データセットは教育・研修寄りの日本語Web文書を再収集・去重して約1.2億テキスト/893億トークンを含み、`sample_10BT`や`small_tokens`など複数サブセットを提供しています。`small_tokens`は最初の1万件が重複するので長期学習には`sample_10BT`を推奨します。citeturn0search0
- `max_train_samples`・`max_eval_samples`を使うとストリーミングを途中で打ち切れるため、RTX3050のような家庭用GPUでも一定トークン数でループできます。
- ストレージ節約のためにもHFのストリーミングAPIを利用していますが、I/O帯域が細いとステップ当たりの速度が落ちるので`HF_DATASETS_CACHE`をSSDに向けるのが無難です。
- ローカルTXTを使う場合は1行=1サンプルで読み込まれ、`tokenizer.eos_token`が自動付加されます。

## 10. トラブルシューティング
| 症状 | 対処 |
| --- | --- |
| `Could not find CUDA` | `torch.version.cuda`とドライバを確認し、CUDA対応wheelをインストール。|
| RamTorch import error | CUDAデバイス未検出時は自動で`nn.Linear`へフォールバックしますが、GPUで実行したい場合は`nvidia-smi`で確認。|
| W&Bで認証エラー | `WANDB_API_KEY`を再セットし、`wandb login`を実行。|
| HFアップロード失敗 | `HF_TOKEN`の権限 (write) と `huggingface_hub`バージョンを確認。|

## 11. 次のステップ
- `configs/train_1b.yaml`を複製して自分用スケジュールを組む。
- eval用分割（例: `c4/en`のvalidation）を別GPUで回したい場合は`logging.eval_interval`を短く調整。
- スケールアウトを検討する場合はRamTorch付属のZeRO実装`ramtorch.zero2`などに乗り換える余地があります。citeturn0search3
