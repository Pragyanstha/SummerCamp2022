# :tropical_fish: SummerCamp2022
[![Evaluate and Send Result](https://github.com/Pragyanstha/SummerCamp2022/actions/workflows/evaluate.yml/badge.svg)](https://github.com/Pragyanstha/SummerCamp2022/actions/workflows/evaluate.yml)
[![Test](https://github.com/Pragyanstha/SummerCamp2022/actions/workflows/tests.yml/badge.svg)](https://github.com/Pragyanstha/SummerCamp2022/actions/workflows/tests.yml)

画像研合宿2022 Team DのRepoです.   
取り組みコンペは[こちら](http://alcon.itlab.org/detail/) 

<img width="867" alt="Screen Shot 2022-08-25 at 23 32 04" src="https://user-images.githubusercontent.com/34847559/186693161-8a0d253a-1601-41df-929f-b092be9b7c5b.png">    

> 上図の5種類の魚を計上するアルゴリズムを開発するコンペです.  


## 構成
```
# tree -L 1
.
├── LICENSE
├── README.md
├── algorithms      # 重要： ここに各種アルゴリズム・モデルのコードがある
├── args.py         # main.pyの受け取る引数を定義 (configファイルの設定に関わる)
├── common          # コード中の共通な処理はここへ
├── configs         # mainのconfigファイルの置き場 (mmdetモデルのconfigファイルもここへ)
├── data            # ドライブからダウンロードしたデータはここに展開
├── data.py         # データローダの定義
├── main.py         # エントリーポイント
├── notebooks       # 実験用ノートブック (新しいことを試した時はぜひこちらで!)
├── requirements.txt
├── results         # 実行結果を出力する場所
├── run.sh          # main.pyを実行するスクリプト
├── tests           # 各種テスト
└── train.py        # モデル学習をする場合に使う
```
学習・検証データ（ドライブにある正解つきテスト画像）は```data/```に展開してください。詳しくは[こちら](data/README.md)へ。

## 環境構築
基本的にはpythonの仮想環境(conda/venv)で作業することをおすすめします.  
手順としてはmmdetをインストールして, その他のパッケージにはrequirements.txtを使います.
### Pytorch
[公式サイト](https://pytorch.org/)からインストールしてください.  
NVIDIAのGPUがある人はCUDA11.3のものを, CPUのみの人はCPU onlyのものを.  

### MMDETのインストール
```
pip install -U openmim
mim install mmcv-full
pip install mmdet
```
### その他パッケージのインステール
```
pip install -r requirements.txt
```
## ベースラインの動作確認
まずは学習済みの重みファイルを```weights/```下に配置します.  
詳しくは[こちら](weights/README.md).  
ダウンロードができたら早速実行じゃ!  
```
python main.py -c configs/baseline.ini
```
すると```results/```下に推論結果とcsvファイルができるはず!!  
できなければorエラーが起きたら[こちら](https://github.com/Pragyanstha/SummerCamp2022/issues/5)のissueにエラー文と一緒にコメントしてくれれば対応します.  

## 開発ルール
ちょっとした開発ルールを設けます. ブランチは主に```develop```から切ってください, developにはみなさんの動くコード(統合版)がmergeされます. ```main```ブランチはアルコンに提出するコードにしましょう.  開発を始める第一歩は以下のコマンド.  
```
git pull
git checkout develop
git checkout -b {新しく作るブランチ名}
```
基本的にはここで作業してもらって, 終わったらremoteにpush(同名のブランチで).  
初回push時はまだremote repositoryにlocalで作成したbranchがないので以下のコマンドでremoteにも作れてlocalの同ブランチをtrackできます.  
```
git push --set-upstream origin {あなたのブランチ名}
```
もちろんこのあとは通用のpushで行けます.  

切りが良いところでdevelopに向けてのプルリクエストを出してください.  
プルリクがmergeされたブランチは消して, 次のタスクは新しく```develop```からブランチを作成しましょう.  
ちなみにブランチ名のネーミングルールを敢えて決めるならこんな感じでしょう.  
```
{あなたの名前}/{タスクの説明}
例) git checkout -b pragyan/update-readme
例) git checkout -b  pragyan/5-create-model # タスクがissueにある場合は先頭にタスク番号入れると良いでしょう.
```
その他のbest practices (個人的なやつ)は[こちら](GUIDELINES.md)にまとめてあります.  

## 学習回す場合
### データセット
学習用のデータセットは[ここ](http://tk2-109-55729.vs.sakura.ne.jp/)から欲しいProblem_*のものをexportする. exportの設定は以下の通り.  
| 項目 | 値 |
----|---- 
| Export format | COCO 1.0 |
| Save images | チェックする |
COCOフォーマット(annotationとimages)で```data/train```に配置.  
```
data/train/
├── annotations
└── images
```

### 学習
```configs/```内にconfigファイルをおけばok.  
configファイルの作り方は```configs/faster_rcnn_r50_fpn.py ```をご参照くださいませ.
```
python train.py configs/faster_rcnn_r50_fpn.py 
```
重みファイルはデフォルトでは```work_dir/```内に作られます. 学習した重みファイルはユニークな名前をつけて```weights/```に保存しましょう (あとは[ドライブ](https://drive.google.com/drive/u/4/folders/1QzUicKbJgSQp-K5CSPSX4jgIue9fppzI)にアップロードしてメンバーに知らせましょう)
アルゴリズムで使うときは```weeights/重みファイル名.pth```パスをconfigで設定します.
### 推論
test_predict.pyの中身みて適当にいじってください.
```
python tests/test_predict.py
```
