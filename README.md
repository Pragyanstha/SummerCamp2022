# :tropical_fish: SummerCamp2022
画像研合宿2022 Team DのRepoです. 
取り組みコンペは[こちら](http://alcon.itlab.org/detail/)  
<img width="867" alt="Screen Shot 2022-08-25 at 23 32 04" src="https://user-images.githubusercontent.com/34847559/186693161-8a0d253a-1601-41df-929f-b092be9b7c5b.png">  

上図の5種類の魚を計上するアルゴリズムを開発するコンペです.  


## 構成
```
# tree -L 1
.                                                                                                          
|-- LICENSE                                                                                                
|-- README.md                                                                                                                         
|-- data                                                                                                   
|-- main.py                                                                                                
|-- tests
|-- requirements.txt
`-- utils
```
学習・検証データ（ドライブにある正解つきテスト画像）は```data/```に展開してください。また、zipファイルは全て展開しておきましょう.  
展開後のデータフォルダの構成です.　　
```
# tree data -L 1
data
|-- Problem_01
|-- Problem_02
|-- Problem_03
|-- Problem_04
|-- output-Problem_01.csv
|-- output-Problem_02.csv
|-- output-Problem_03.csv
`-- output-Problem_04.csv
```

## データセットの配置
COCOフォーマット(annotationとimages)を```data/train```に配置  
```
data/train/
|-- annotations
`-- images
```

## MMDETのインストール
```
pip install -U openmim
mim install mmcv-full
pip install mmdet
```
## その他パッケージのインステール
```
pip install -r requirements.txt
```

## 学習
```configs/```内にconfigファイルをおけばok.  
configファイルの作り方は```configs/faster_rcnn_r50_fpn.py ```をご参照くださいませ.
```
python train.py configs/faster_rcnn_r50_fpn.py 
```

## 推論
predict.pyの中身みて適当にいじってください.
```
python predict.py
```
