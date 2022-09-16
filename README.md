# :tropical_fish: SummerCamp2022
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


## 学習する場合
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
