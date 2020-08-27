## 训练
环境配置：python3.6 + pytorch0.4.1，更详细的环境配置内容及可能遇到的bug，可参考[PPDM](https://km.sankuai.com/space/~fengqi06 "PPDM")。

公开数据集HICO-DET上的训练：


    cd src
    python main.py  hoidet --batch_size 32 --lr 5e-4 --gpus 0,1  --num_workers 16  --load_model ../models/ctdet_coco_dla_2x.pth --image_dir images/train2015 --dataset hico --exp_id hoidet_hico_dla

说明：
--load_model：选择的模型backbone，可选择（Res18，DLA34，Hourglass104）, 模型放在models文件夹下。其中Hourglass104效果最佳；在使用DLA34的时候发现测试速度很慢，可尝试升级至pytorch1.0
-- exp_id：训练结果保存地址
--image_dir：数据集文件名，里面包含所有的图片，放在Dataset文件夹下

酒店数据集上的训练：
1、


    cd src
    python main.py  hoidet --batch_size 16 --lr 1e-3 --gpus 0,1  --num_workers 8  --load_model ../models/exdet_hg.pth --image_dir Hotel_20200509_images_v1 --dataset hoia --exp_id hotelcleaning_hg_v2/data_v2_hoi --lr_step '40,50' --num_epochs 60 --arch hourglass --train_data train_hotel_v2.json
酒店数据集摆放：Dataset/Hotelcleaning/Hotel_20200509_images_v1, Dataset/Hotelcleaning/annotations

2、在训练时，同时每个epoch做一次test，保存best model


    cd src1
    python main.py  hoidet --batch_size 16 --lr 1e-3 --gpus 0,1  --num_workers 8  --load_model ../../PPDM-master2/exp/hoidet/hotelcleaning_hg_v2/data_v2_detection/model_best.pth --image_dir Hotel_20200509_images_v1 --dataset hoia --exp_id hotelcleaning_hg_v2/data_v2_pretrain_hoi/model3 --lr_step '40,50' --num_epochs 60 --arch hourglass --train_data train_hotel_v2.json --test_data test_hotel_v2.json --test_with_eval
3、只训练模型的目标检测，生成预训练model


    cd src2
    python main.py  hoidet --batch_size 16 --lr 1e-3 --gpus 0,1  --num_workers 8  --load_model ../models/exdet_hg.pth --image_dir Hotel_20200509_images_v1 --dataset hoia --exp_id hotelcleaning_hg_v2/data_v2_hoi --lr_step '40,50' --num_epochs 60 --arch hourglass --train_data train_hotel_allbox_v1.json
## 测试
酒店数据测试：


    cd src
    python test_hoi.py hoidet --exp_id hotelcleaning_hg_v2/data_v2_pretrain_hoi --load_model ../../PPDM-master/exp/hoidet/hotelcleaning_hg_v2/data_v2_pretrain_hoi/model_best.pth --gpus 0 --dataset hoia --image_dir Hotel_20200509_images_v1 --arch hourglass --test_with_eval --test_data test_hotel_v2.json
--test_data：测试数据集json文件
