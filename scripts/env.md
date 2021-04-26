



# 1 . 参考资料






# 2. 安装方法

## 2.1 


docker_horizon_xj3_tc_v1.1.21i
    export version=... # 写成实际的版本号
    gunzip -c docker_horizon_xj3_tc_v1.1.21i.tar.gz | docker load


docker 权限问题，先解决这个

https://blog.csdn.net/weixin_47758362/article/details/108050348

# jcq @ jcq-linux in /media/jcq/Work/Horizan/0309/算法工具链/1.1.21i_en [10:07:23] 
$ sudo gunzip -c docker_horizon_xj3_tc_v1.1.21i.tar.gz | docker load
Loaded image: docker.hobot.cc/aitools/horizon_xj3_tc:xj3_1.1.21i


## 2.2 

sudo docker run -it --rm -v /media/jcq/Work/Horizan/x3:/data docker.hobot.cc/aitools/horizon_xj3_tc:xj3_1.1.21i


>
# jcq @ jcq-linux in /media/jcq/Work/Horizan/0309/算法工具链/1.1.21i_en [10:07:38] 
$ sudo docker run -it --rm -v /media/jcq/Work/Horizan/x3:/data docker.hobot.cc/aitools/horizon_xj3_tc:xj3_1.1.21i
[root@0cfaf1b9be09 /]# 



# 3 . 实操

[root@0cfaf1b9be09 01_lenet_gray]# pwd
/data/samples/05_miscellaneous/01_lenet_gray

进入 mapper

[root@0cfaf1b9be09 mapper]# ls
01_check.sh      03_build.sh      inference.py            process_mnist.py
02_get_mnist.sh  04_inference.sh  lenet_gray_config.yaml  README.cn.md



记得 env.conf 中对于路径的设置为：

proto="/data/modelzoo/mapper/other/lenet/lenet_train_test.prototxt"
caffe_model="/data/modelzoo/mapper/other/lenet/lenet_iter_100000.caffemodel"

因为是2个独立的系统，不能用host 路径来设置路径



[root@0cfaf1b9be09 mapper]# ./01_check.sh 
++ dirname ./01_check.sh
+ cd .
+ . ../env.conf
++ sample_name=lenet_gray
++ input_width=28
++ input_height=28
++ input_type=0
++ score_threshold=0
++ model_type=caffe
++ proto=/data/modelzoo/mapper/other/lenet/lenet_train_test.prototxt
++ caffe_model=/data/modelzoo/mapper/other/lenet/lenet_iter_100000.caffemodel
++ test_image=../lenet_data/0.jpg
++ board_test_image=/userdata/samples/lenet_data/0.jpg
+ output=./lenet_gray_checker.log
+ march=bernoulli2
+ hb_mapper checker --model-type caffe --proto /data/modelzoo/mapper/other/lenet/lenet_train_test.prototxt --model /data/modelzoo/mapper/other/lenet/lenet_iter_100000.caffemodel --output ./lenet_gray_checker.log --march bernoulli2
2021-04-20 15:30:28,644 INFO Start hb_mapper....
2021-04-20 15:30:28,644 INFO hb_mapper version 1.1.58
2021-04-20 15:30:28,647 INFO Model type: caffe
2021-04-20 15:30:28,648 INFO output file: ./lenet_gray_checker.log
2021-04-20 15:30:28,648 INFO input names []
2021-04-20 15:30:28,648 INFO input shapes {}
2021-04-20 15:30:28,648 INFO Begin model checking....
2021-04-20 15:30:28,649 INFO [Tue Apr 20 15:30:28 2021] Start to Horizon NN Model Convert.
2021-04-20 15:30:28,649 INFO The input parameter is not specified, convert with default parameters.
2021-04-20 15:30:28,649 INFO The hbdk parameter is not specified, and the submodel will be compiled with the default parameter.
2021-04-20 15:30:28,649 INFO HorizonNN version: 0.9.7
2021-04-20 15:30:28,649 INFO HBDK version: 3.16.6
2021-04-20 15:30:28,649 INFO [Tue Apr 20 15:30:28 2021] Start to parse the caffe model.
2021-04-20 15:30:28,653 INFO Find 1 inputs in the model:
2021-04-20 15:30:28,653 INFO Got input 'data' with shape [1, 1, 28, 28].
2021-04-20 15:30:28,754 INFO [Tue Apr 20 15:30:28 2021] End to parse the caffe model.
2021-04-20 15:30:28,755 INFO Model input names: ['data']
2021-04-20 15:30:28,760 INFO Saving the original float model: ./.hb_check/original_float_model.onnx.
2021-04-20 15:30:28,760 INFO [Tue Apr 20 15:30:28 2021] Start to optimize the model.
2021-04-20 15:30:28,919 INFO [Tue Apr 20 15:30:28 2021] End to optimize the model.
2021-04-20 15:30:28,926 INFO Saving the optimized model: ./.hb_check/optimized_float_model.onnx.
2021-04-20 15:30:28,926 INFO [Tue Apr 20 15:30:28 2021] Start to calibrate the model.
2021-04-20 15:30:28,929 INFO [Tue Apr 20 15:30:28 2021] End to calibrate the model.
2021-04-20 15:30:28,929 INFO [Tue Apr 20 15:30:28 2021] Start to quantize the model.
2021-04-20 15:30:28,964 INFO [Tue Apr 20 15:30:28 2021] End to quantize the model.
2021-04-20 15:30:28,990 INFO Saving the quantized model: ./.hb_check/quantized_model.onnx.
2021-04-20 15:30:28,990 INFO [Tue Apr 20 15:30:28 2021] Start to compile the model with march bernoulli2.
2021-04-20 15:30:29,024 INFO Compile submodel: LeNet_subgraph_0
2021-04-20 15:30:29,057 INFO hbdk-cc parameters:{'optimize-level': 'O0', 'input-source': 'ddr', 'input-layout': 'NHWC', 'output-layout': 'NCHW'}
[==================================================] 100%
2021-04-20 15:30:29,175 INFO [Tue Apr 20 15:30:29 2021] End to compile the model with march bernoulli2.
2021-04-20 15:30:29,175 INFO The converted model node information:
==================================================
Node           ON   Subgraph  Type                
--------------------------------------------------
conv1          BPU  id(0)     HzSQuantizedConv    
pool1          BPU  id(0)     HzQuantizedMaxPool  
conv2          BPU  id(0)     HzSQuantizedConv    
pool2          BPU  id(0)     HzQuantizedMaxPool  
ip1            BPU  id(0)     HzSQuantizedConv    
ip2            BPU  id(0)     HzSQuantizedConv    
ip2_reshape_0  CPU  --        Reshape             
2021-04-20 15:30:29,176 INFO [Tue Apr 20 15:30:29 2021] End to Horizon NN Model Convert.
2021-04-20 15:30:29,176 INFO model deps info empty
2021-04-20 15:30:29,180 INFO End model checking....
[root@0cfaf1b9be09 mapper]# 





[root@0cfaf1b9be09 mapper]# python3 process_mnist.py t10k-images-idx3-ubyte 

[root@0cfaf1b9be09 mapper]# python3 process_mnist.py t10k-images-idx3-ubyte 
write file: mnist_cal_data/0.bin
write file: mnist_cal_data/1.bin
write file: mnist_cal_data/2.bin
write file: mnist_cal_data/3.bin
write file: mnist_cal_data/4.bin
write file: mnist_cal_data/5.bin
write file: mnist_cal_data/6.bin
write file: mnist_cal_data/7.bin
write file: mnist_cal_data/8.bin
write file: mnist_cal_data/9.bin
write file: mnist_cal_data/10.bin
write file: mnist_cal_data/11.bin
write file: mnist_cal_data/12.bin
write file: mnist_cal_data/13.bin
write file: mnist_cal_data/14.bin
write file: mnist_cal_data/15.bin
write file: mnist_cal_data/16.bin
write file: mnist_cal_data/17.bin
write file: mnist_cal_data/18.bin
write file: mnist_cal_data/19.bin
write file: mnist_cal_data/20.bin
write file: mnist_cal_data/21.bin
write file: mnist_cal_data/22.bin
write file: mnist_cal_data/23.bin
write file: mnist_cal_data/24.bin
write file: mnist_cal_data/25.bin
write file: mnist_cal_data/26.bin
write file: mnist_cal_data/27.bin
write file: mnist_cal_data/28.bin
write file: mnist_cal_data/29.bin
write file: mnist_cal_data/30.bin
write file: mnist_cal_data/31.bin
write file: mnist_cal_data/32.bin
write file: mnist_cal_data/33.bin
write file: mnist_cal_data/34.bin
write file: mnist_cal_data/35.bin
write file: mnist_cal_data/36.bin
write file: mnist_cal_data/37.bin
write file: mnist_cal_data/38.bin
write file: mnist_cal_data/39.bin
write file: mnist_cal_data/40.bin
write file: mnist_cal_data/41.bin
write file: mnist_cal_data/42.bin
write file: mnist_cal_data/43.bin
write file: mnist_cal_data/44.bin
write file: mnist_cal_data/45.bin
write file: mnist_cal_data/46.bin
write file: mnist_cal_data/47.bin
write file: mnist_cal_data/48.bin
write file: mnist_cal_data/49.bin
write file: mnist_cal_data/50.bin
write file: mnist_cal_data/51.bin
write file: mnist_cal_data/52.bin
write file: mnist_cal_data/53.bin
write file: mnist_cal_data/54.bin
write file: mnist_cal_data/55.bin
write file: mnist_cal_data/56.bin
write file: mnist_cal_data/57.bin
write file: mnist_cal_data/58.bin
write file: mnist_cal_data/59.bin
write file: mnist_cal_data/60.bin
write file: mnist_cal_data/61.bin
write file: mnist_cal_data/62.bin
write file: mnist_cal_data/63.bin
write file: mnist_cal_data/64.bin
write file: mnist_cal_data/65.bin
write file: mnist_cal_data/66.bin
write file: mnist_cal_data/67.bin
write file: mnist_cal_data/68.bin
write file: mnist_cal_data/69.bin
write file: mnist_cal_data/70.bin
write file: mnist_cal_data/71.bin
write file: mnist_cal_data/72.bin
write file: mnist_cal_data/73.bin
write file: mnist_cal_data/74.bin
write file: mnist_cal_data/75.bin
write file: mnist_cal_data/76.bin
write file: mnist_cal_data/77.bin
write file: mnist_cal_data/78.bin
write file: mnist_cal_data/79.bin
write file: mnist_cal_data/80.bin
write file: mnist_cal_data/81.bin
write file: mnist_cal_data/82.bin
write file: mnist_cal_data/83.bin
write file: mnist_cal_data/84.bin
write file: mnist_cal_data/85.bin
write file: mnist_cal_data/86.bin
write file: mnist_cal_data/87.bin
write file: mnist_cal_data/88.bin
write file: mnist_cal_data/89.bin
write file: mnist_cal_data/90.bin
write file: mnist_cal_data/91.bin
write file: mnist_cal_data/92.bin
write file: mnist_cal_data/93.bin
write file: mnist_cal_data/94.bin
write file: mnist_cal_data/95.bin
write file: mnist_cal_data/96.bin
write file: mnist_cal_data/97.bin
write file: mnist_cal_data/98.bin
write file: mnist_cal_data/99.bin



## build.sh 

[root@0cfaf1b9be09 mapper]# ./03_build.sh 

cd $(dirname $0) || exit
. ../env.conf
sample_name='lenet_gray'
input_width=28
input_height=28
input_type=0  # BPU_TYPE_IMG_Y
score_threshold=0


model_type="caffe"
proto="/data/modelzoo/mapper/other/lenet/lenet_train_test.prototxt"
caffe_model="/data/modelzoo/mapper/other/lenet/lenet_iter_100000.caffemodel"

#proto="./lenet_train_test.prototxt"
#caffe_model="./lenet_iter_100000.caffemodel"

test_image='../lenet_data/0.jpg'
board_test_image='/userdata/samples/lenet_data/0.jpg'


config_file="./${sample_name}_config.yaml"
model_type="caffe"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
2021-04-20 17:06:18,797 INFO Start hb_mapper....
2021-04-20 17:06:18,797 INFO hb_mapper version 1.1.58
2021-04-20 17:06:18,810 INFO norm_types[i]: no_preprocess
2021-04-20 17:06:18,814 INFO Working dir: /data/samples/05_miscellaneous/01_lenet_gray/mapper/model_output
2021-04-20 17:06:18,814 INFO Start Model Convert....
2021-04-20 17:06:18,817 INFO First calibration picture name: 0.bin
2021-04-20 17:06:18,817 INFO call build params:
 {'march': 'bernoulli2', 'debug_mode': False, 'save_model': True, 'name_prefix': 'lenet_gray', 'input_dict': {'data': {'input_shape': [1, 1, 28, 28], 'expected_input_type': 'GRAY_128', 'original_input_type': 'GRAY'}}, 'cali_dict': {'calibration_type': 'kl', 'calibration_loader': {'data': <horizon_nn.data.loader.TransformLoader object at 0x7feb0fa44080>}, 'per_channel': False, 'max_percentile': 1.0}, 'hbdk_dict': {'compile_mode': 'latency', 'debug': False, 'optimize_level': 'O3', 'input_source': {'data': 'pyramid'}}, 'node_dict': {}}
2021-04-20 17:06:18,828 INFO [Tue Apr 20 17:06:18 2021] Start to Horizon NN Model Convert.
2021-04-20 17:06:18,828 INFO Parsing the input parameter:{'data': {'input_shape': [1, 1, 28, 28], 'expected_input_type': 'GRAY_128', 'original_input_type': 'GRAY'}}
2021-04-20 17:06:18,828 INFO Parsing the calibration parameter
2021-04-20 17:06:18,848 INFO Parsing the hbdk parameter:{'compile_mode': 'latency', 'debug': False, 'optimize_level': 'O3', 'input_source': {'data': 'pyramid'}}
2021-04-20 17:06:18,848 INFO HorizonNN version: 0.9.7
2021-04-20 17:06:18,848 INFO HBDK version: 3.16.6
2021-04-20 17:06:18,848 INFO [Tue Apr 20 17:06:18 2021] Start to parse the caffe model.
2021-04-20 17:06:18,851 INFO Find 1 inputs in the model:
2021-04-20 17:06:18,851 INFO Got input 'data' with shape [1, 1, 28, 28].
2021-04-20 17:06:18,948 INFO [Tue Apr 20 17:06:18 2021] End to parse the caffe model.
2021-04-20 17:06:18,949 INFO Model input names: ['data']
2021-04-20 17:06:18,949 INFO Create a preprocessing operator for input_name data with means=None, std=None, original_input_layout=NCHW, color convert from 'GRAY' to 'GRAY'.
2021-04-20 17:06:18,957 INFO Saving the original float model: lenet_gray_original_float_model.onnx.
2021-04-20 17:06:18,957 INFO [Tue Apr 20 17:06:18 2021] Start to optimize the model.
2021-04-20 17:06:19,076 INFO [Tue Apr 20 17:06:19 2021] End to optimize the model.
2021-04-20 17:06:19,083 INFO Saving the optimized model: lenet_gray_optimized_float_model.onnx.
2021-04-20 17:06:19,083 INFO [Tue Apr 20 17:06:19 2021] Start to calibrate the model.
2021-04-20 17:06:19,085 INFO number of calibration data samples: 100
2021-04-20 17:06:19,086 INFO Run calibration model with kl method.
2021-04-20 17:06:19,946 INFO [Tue Apr 20 17:06:19 2021] End to calibrate the model.
2021-04-20 17:06:19,947 INFO [Tue Apr 20 17:06:19 2021] Start to quantize the model.
2021-04-20 17:06:20,022 INFO [Tue Apr 20 17:06:20 2021] End to quantize the model.
2021-04-20 17:06:20,049 INFO Saving the quantized model: lenet_gray_quantized_model.onnx.
2021-04-20 17:06:20,049 INFO [Tue Apr 20 17:06:20 2021] Start to compile the model with march bernoulli2.
2021-04-20 17:06:20,076 INFO Compile submodel: LeNet_subgraph_0
2021-04-20 17:06:20,109 INFO hbdk-cc parameters:{'optimize-level': 'O3', 'input-source': 'pyramid', 'optimize-target': 'fast', 'input-layout': 'NHWC', 'output-layout': 'NCHW'}
[==================================================] 100%
2021-04-20 17:06:20,339 INFO [Tue Apr 20 17:06:20 2021] End to compile the model with march bernoulli2.
2021-04-20 17:06:20,339 INFO The converted model node information:
==================================================================================================================
Node                    ON   Subgraph  Type                    Cosine Similarity  Threshold                       
------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_data  BPU  id(0)     HzSQuantizedPreprocess  0.999995           127.000000                      
conv1                   BPU  id(0)     HzSQuantizedConv        0.999909           253.879395                      
pool1                   BPU  id(0)     HzQuantizedMaxPool      0.999921           1405.619019                     
conv2                   BPU  id(0)     HzSQuantizedConv        0.999893           1405.619019                     
pool2                   BPU  id(0)     HzQuantizedMaxPool      0.999876           4083.379395                     
ip1                     BPU  id(0)     HzSQuantizedConv        0.999910           4083.379395                     
ip2                     BPU  id(0)     HzSQuantizedConv        0.999989           2438.901855                     
ip2_reshape_0           CPU  --        Reshape                 
2021-04-20 17:06:20,339 INFO The quantify model output:
=======================================================================
Node  Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-----------------------------------------------------------------------
ip2   0.999989           7.732184     2.969808     16.220459           
2021-04-20 17:06:20,340 INFO [Tue Apr 20 17:06:20 2021] End to Horizon NN Model Convert.
2021-04-20 17:06:20,340 INFO start convert to *.bin file....
2021-04-20 17:06:20,363 INFO ########################################
2021-04-20 17:06:20,363 INFO ----------- dependency info ------------
2021-04-20 17:06:20,363 INFO hb mapper version: 1.1.58
2021-04-20 17:06:20,363 INFO hbdk version: 3.16.6
2021-04-20 17:06:20,363 INFO hbdk runtime version: 3.10.8
2021-04-20 17:06:20,363 INFO horizon_nn version: 0.9.7
2021-04-20 17:06:20,363 INFO -------- model parameters info ---------
2021-04-20 17:06:20,363 INFO caffe_model: /data/modelzoo/mapper/other/lenet/lenet_iter_100000.caffemodel
2021-04-20 17:06:20,363 INFO prototxt: /data/modelzoo/mapper/other/lenet/lenet_train_test.prototxt
2021-04-20 17:06:20,363 INFO onnx_model: 
2021-04-20 17:06:20,363 INFO layer_out_dump: False
2021-04-20 17:06:20,364 INFO output_layout: None
2021-04-20 17:06:20,364 INFO -------- input_parameters info ---------
2021-04-20 17:06:20,364 INFO -------- input info : data -------
2021-04-20 17:06:20,364 INFO --input_name        : data
2021-04-20 17:06:20,364 INFO --input_type_rt    : gray
2021-04-20 17:06:20,364 INFO --input_type_train : gray
2021-04-20 17:06:20,364 INFO --norm_type        : no_preprocess
2021-04-20 17:06:20,364 INFO --input_shape      : 1x1x28x28
2021-04-20 17:06:20,364 INFO ----------------------------------
2021-04-20 17:06:20,364 INFO -------- calibration parameters info ---------
2021-04-20 17:06:20,364 INFO preprocess_on: False
2021-04-20 17:06:20,364 INFO calibration_type: kl
2021-04-20 17:06:20,364 INFO per_channel: False
2021-04-20 17:06:20,364 INFO max_percentile: 1.0
2021-04-20 17:06:20,365 INFO ------------ compiler_parameters info -------------
2021-04-20 17:06:20,365 INFO compile_mode: latency
2021-04-20 17:06:20,365 INFO debug: False
2021-04-20 17:06:20,365 INFO optimize_level: O3
2021-04-20 17:06:20,365 INFO input_source: {'data': 'pyramid'}
2021-04-20 17:06:20,365 INFO ########################################
2021-04-20 17:06:20,368 INFO Convert to runtime bin file sucessfully!
2021-04-20 17:06:20,368 INFO End Model Convert




## inference 
[root@0cfaf1b9be09 mapper]# ./04_inference.sh 
prob: [[0 2 7 6 5 1 3 8 9 4]]





# 三.onnx 转 定点模型.

 
## 3.1 


sudo docker run -it --rm -v /media/jcq/Work/Horizan/x3:/data docker.hobot.cc/aitools/horizon_xj3_tc:xj3_1.1.21i


cd /data/samples/05_miscellaneous/01_lenet_gray/mapper


准备onnx模型，以PyTorch 为例，训练LeNet, 保存训练模型output/mnist.pth， 转换为 output/mnist.onnx

详情请参照: https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
或者直接下载 PytorchTensorflowMnist.ipynb: https://pan.horizon.ai/index.php/s/ESHHfL4EZk97CxL
其他框架如 mxnet, tensorflow, cntk, pytorch等，请参考开源工程: https://github.com/onnx/tutorials/tree/master/tutorials



## 3.2 sh 01_check.sh

使用 lenet.onnx 文件做测试


得到mnist.onnx后，浮点转换流程与 caffe转定点模型类似.

修改01_check.sh: model-type为onnx, proto和 model 为mnist.onnx


会遇到不支持的Op情况，

需要替换 return F.log_softmax(x, dim=1) 为 return x



### 3.2.1 onnx version 

将保持的onnx opset 设置为 10，即可进行onnx 模型转换


[root@b6268e9fc0be mapper]# sh 01_check.sh 
++ dirname 01_check.sh
+ cd .
+ . ../env.conf
++ sample_name=lenet_gray
++ input_width=28
++ input_height=28
++ input_type=0
++ score_threshold=0
++ model_type=onnx
++ proto=/data/modelzoo/mapper/other/lenet/lenet.onnx
++ caffe_model=/data/modelzoo/mapper/other/lenet/lenet.onnx
++ test_image=../lenet_data/0.jpg
++ board_test_image=/userdata/samples/lenet_data/0.jpg
+ output=./lenet_gray_checker.log
+ march=bernoulli2
+ hb_mapper checker --model-type onnx --proto /data/modelzoo/mapper/other/lenet/lenet.onnx --model /data/modelzoo/mapper/other/lenet/lenet.onnx --output ./lenet_gray_checker.log --march bernoulli2
2021-04-26 10:09:52,103 INFO Start hb_mapper....
2021-04-26 10:09:52,103 INFO hb_mapper version 1.1.58
2021-04-26 10:09:52,104 INFO Model type: onnx
2021-04-26 10:09:52,105 INFO output file: ./lenet_gray_checker.log
2021-04-26 10:09:52,105 INFO input names []
2021-04-26 10:09:52,105 INFO input shapes {}
2021-04-26 10:09:52,105 INFO Begin model checking....
2021-04-26 10:09:52,105 INFO [Mon Apr 26 10:09:52 2021] Start to Horizon NN Model Convert.
2021-04-26 10:09:52,105 INFO The input parameter is not specified, convert with default parameters.
2021-04-26 10:09:52,105 INFO The hbdk parameter is not specified, and the submodel will be compiled with the default parameter.
2021-04-26 10:09:52,105 INFO HorizonNN version: 0.9.7
2021-04-26 10:09:52,106 INFO HBDK version: 3.16.6
2021-04-26 10:09:52,106 INFO [Mon Apr 26 10:09:52 2021] Start to parse the onnx model.
2021-04-26 10:09:52,111 INFO ONNX model info:
ONNX IR version:  4
Opset version:    10
Input name:       input.1, [1, 3, 32, 32]
2021-04-26 10:09:52,113 INFO [Mon Apr 26 10:09:52 2021] End to parse the onnx model.
2021-04-26 10:09:52,113 INFO Model input names: ['input.1']
2021-04-26 10:09:52,114 INFO Saving the original float model: ./.hb_check/original_float_model.onnx.
2021-04-26 10:09:52,115 INFO [Mon Apr 26 10:09:52 2021] Start to optimize the model.
2021-04-26 10:09:52,229 INFO [Mon Apr 26 10:09:52 2021] End to optimize the model.
2021-04-26 10:09:52,231 INFO Saving the optimized model: ./.hb_check/optimized_float_model.onnx.
2021-04-26 10:09:52,231 INFO [Mon Apr 26 10:09:52 2021] Start to calibrate the model.
2021-04-26 10:09:52,233 INFO [Mon Apr 26 10:09:52 2021] End to calibrate the model.
2021-04-26 10:09:52,233 INFO [Mon Apr 26 10:09:52 2021] Start to quantize the model.
2021-04-26 10:09:52,239 INFO [Mon Apr 26 10:09:52 2021] End to quantize the model.
2021-04-26 10:09:52,244 INFO Saving the quantized model: ./.hb_check/quantized_model.onnx.
2021-04-26 10:09:52,244 INFO [Mon Apr 26 10:09:52 2021] Start to compile the model with march bernoulli2.
2021-04-26 10:09:52,253 INFO Compile submodel: torch-jit-export_subgraph_0
WARNING: the onnx model's ir_version is inconsistent with the ir_version of the parser.
2021-04-26 10:09:52,259 INFO hbdk-cc parameters:{'optimize-level': 'O0', 'input-source': 'ddr', 'input-layout': 'NHWC', 'output-layout': 'NCHW'}
[==================================================] 100%
2021-04-26 10:09:52,309 INFO [Mon Apr 26 10:09:52 2021] End to compile the model with march bernoulli2.
2021-04-26 10:09:52,309 INFO The converted model node information:
=================================================
Node          ON   Subgraph  Type                
-------------------------------------------------
11            BPU  id(0)     HzSQuantizedConv    
13            BPU  id(0)     HzQuantizedMaxPool  
14            BPU  id(0)     HzSQuantizedConv    
16            BPU  id(0)     HzQuantizedMaxPool  
25            BPU  id(0)     HzSQuantizedConv    
27            BPU  id(0)     HzSQuantizedConv    
29            BPU  id(0)     HzSQuantizedConv    
29_reshape_0  CPU  --        Reshape             
2021-04-26 10:09:52,310 INFO [Mon Apr 26 10:09:52 2021] End to Horizon NN Model Convert.
2021-04-26 10:09:52,310 INFO model deps info empty
2021-04-26 10:09:52,312 INFO End model checking....

02_get_mnist.sh: 不做修改



## 03. 03_build.sh:

model_type="onnx"

修改lenet_gray_config.yaml

[root@b6268e9fc0be mapper]# sh 03_build.sh 

cd $(dirname $0) || exit
. ../env.conf
sample_name='lenet_gray'
input_width=28
input_height=28
input_type=0  # BPU_TYPE_IMG_Y
score_threshold=0


model_type="onnx"
# proto="/data/modelzoo/mapper/other/lenet/mnist-8.onnx"

# caffe_model="/data/modelzoo/mapper/other/lenet/mnist-8.onnx"

proto="/data/modelzoo/mapper/other/lenet/lenet.onnx"

caffe_model="/data/modelzoo/mapper/other/lenet/lenet.onnx"


test_image='../lenet_data/0.jpg'
board_test_image='/userdata/samples/lenet_data/0.jpg'


config_file="./${sample_name}_config.yaml"
# model_type="caffe"
model_type="onnx"  
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
2021-04-26 10:13:59,201 INFO Start hb_mapper....
2021-04-26 10:13:59,201 INFO hb_mapper version 1.1.58
2021-04-26 10:13:59,215 INFO norm_types[i]: no_preprocess
2021-04-26 10:13:59,219 INFO Working dir: /data/samples/05_miscellaneous/01_lenet_gray/mapper/model_output
2021-04-26 10:13:59,219 INFO Start Model Convert....
2021-04-26 10:13:59,220 ERROR wrong input name: 'data', available: ['input.1']
2021-04-26 10:13:59,220 ERROR check_input_names failed


这里输入的数据集并非是 mnist ，而是



04_inference.sh : 不做修改







# 四.定点模型上板测试.

4.1， runtime_arm 为上板测试文件夹

01_build.sh: 编译infer程序, 打包相关依赖库.

源码为: samples\02_rt_sample_src\01_hr_example\src\infer.cc


将测试图片../lenet_data/0.jpg，runtime_arm 文件夹，以及../mapper/model_output/lenet_gray_hybrid_horizonrt.bin

拷贝到开发板的/userdata/

修改dev_board_infer.sh


运行结果:


4.2， runtime_sim 为PC端模拟器测试

01_build.sh :


02_infer.sh









# 五, 测试mobilenetV2推理速度

## 5.1 文件功能说明


sudo docker run -it --rm -v /media/jcq/Work/Horizan/x3:/data docker.hobot.cc/aitools/horizon_xj3_tc:xj3_1.1.21i


cd /data/samples/03_classification/01_mobilenet/mapper


env.conf: 配置示例名称 sample_name=mobilenetv2

mapper: 将浮点模型转换为定点模型

runtime_arm: 上板测试

runtime_sim: 模拟器测试

## 5.2 mapper编译模型

检查模型: sh 01_check.sh

修改路径 ：
../../../01_common/modelzoo  --> /data/modelzoo


放入原始图片到: ../../../01_common/data/imagenet/calibration_data




校准数据预处理: sh 02_preprocess.sh

复制yaml: cp mobilenet_config.yaml mobilenetv2_config.yaml

修改mobilenetv2_config.yaml

# Caffe浮点网络数据模型文件
caffe_model: '../../../01_common/modelzoo/mapper/classification/mobilenet/mobilenet_v2.caffemodel'
# Caffe网络描述文件
prototxt: '../../../01_common/modelzoo/mapper/classification/mobilenet/mobilenet_v2_deploy.prototxt'
# 模型转换输出的用于上板执行的模型文件
output_model_file_prefix: 'mobilenetv2'
修改 03_build.sh： config_file="./mobilenetv2_config.yaml"

编译模型: sh 03_build.sh

编译成功后，runtime使用模型为: model_output/mobilenetv2_hybrid_horizonrt.bin

修改04_inference.sh : model_file="./model_output/mobilenetv2_quantized_model.onnx"

运行单图测试: sh 04_inference.sh

5.3 runtime_arm 上板跑FPS【单核单帧】

登陆开发板: ssh root@192.168.168.10 .

固定CPU和BPU频率:

echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
export HR_NODE_PROFILER=true
进入文件夹: cd runtime_arm

编译源码: sh 01_build.sh

数据预处理: sh 02_preprocess.sh

将数据和脚本发送到开发板上的/userdata/samples/:

sh 03_scp_to_board.sh 192.168.168.10

跑帧率:

sh 05_perf.sh 192.168.168.10

运行结果:

===REPORT-START{RUNTIME-ARM-PERF}===

Whole process statistics:count:100, duration:298.028ms, min:2.765ms, max:4.475ms, average:2.96722ms, fps:335.539/s

, Infer stage statistics:count:100, duration:296.515ms, min:2.751ms, max:4.456ms, average:2.95212ms, fps:337.251/s

, Post process stage statistics:count:100, duration:1.376ms, min:0.012ms, max:0.056ms, average:0.0133469ms, fps:72674.4/s

===REPORT-END{RUNTIME-ARM-PERF}===

测试结果解读:

Whole process: 模型推理耗时+后处理耗时，

最短耗时:2.765ms, 最长耗时:4.475ms, 平均耗时:2.96722ms, 帧率为 335.539 FPS

Infer stage: 模型推理耗时统计

Post process stage: 后处理耗时统计

5.4 运行【双核双帧】测试

登陆开发板: ssh root@192.168.168.10 .

固定CPU和BPU频率:

echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
export HR_NODE_PROFILER=true
修改配置: mobilenetv2_config.yaml

input_type_rt: 'nv12'

编译模型: sh 03_build.sh

进入双帧双核测试目录:

cd samples/05_miscellaneous/06_dual_frame_dual_core/shell

编译测试源码:

注释掉: #export LINARO_GCC_ROOT=

sh build.sh

拷贝定点模型到shell目录:

cp ../../../03_classification/01_mobilenet/mapper/model_output/mobilenetv2_hybrid_horizonrt.bin ./

将shell文件夹拷贝到开发板/userdata/samples:

scp -r ../shell root@192.168.168.10:/userdata/samples/

开始测试, 开发板上执行:

sh start_mobileNetV2.sh

运行一段时间，终止程序, 查看Log:

method name:PostProcessMethod time:0.146

frame_rate: 672.438

测试结果解读:

PostProcessMethod: 后处理耗时，0.146ms

frame_rate: 帧率 672.438 FPS, 每个BPU核独立跑一帧 为 672.438/2 = 336.219 FPS







