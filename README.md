# GANS-With-FasterRcnn
> data_enhancement

## Dep

* pytorch v0.3
* `in output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_70000.pth` - is pretrained model
* `/output/res101/voc_2007_trainval/default/epoch_2_TRAIN_DETECT_netM.pth` - is pretrain netg model

## Tips

* val your code - use`exprements/scripts/test_faster_rcnn.sh` - **not test_net.py**

## TODO

* [x] - Run this code
  * input =  [375, 500]
  * output = [38, 50]
* [x] - add netG mask part / `in lib/nets/masknet.py`
* [x] - add netD part / `in lib/nets/masknet.py`
* [x] - create train entrypoint file **base on train_val file**
* [x] - complete full train phase / `from iter 70000 to 14000, save it in output/default/voc_2007_trainval/res1010_faster_rcnn_iter_70000.pth(means iter 70000)`
* [ ] - val this `pth` file

## Changelog

* ~~add netG&D optimizer into origin~~
* just upsample in netG(be same size as origin image), not by rconv
* **change pretrainmodel name by add prefix pre_**