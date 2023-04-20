![GithubHeader](https://user-images.githubusercontent.com/37477845/92315782-e1255d80-f025-11ea-80e0-e62fc08c7a1e.gif)

# CudaOpenCVObjectDetectionDemo
OpenCV Video Object Detection with Cuda Accelerated.
- CPU Detection Speed: about 30 fps on Intel i9 
- GPU Detection Speed: about 100 fps on NVIDIA RTX 3060

<img width=800 height=480 src="https://github.com/Think-Big-Do-Small/CudaOpenCVObjectDetectionDemo/blob/9f1ac74cc5ec119d76ef93df82242b238fa0ef66/screenshot.png"></img>

```bash
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace std; 
using namespace cv; 
using namespace dnn; 

int main(int argc, char**)
{
	string file_path = "./cmake_tutorial/cvModels/mobilenet_V2/";
	vector<string>class_names;
	ifstream ifs(string(file_path + "object_detection_classes_coco.txt").c_str());
	string line;

	// Load in all the classes from the file 
	while (getline(ifs, line)){
		cout << line << endl;
		class_names.push_back(line);
	}

	// Read in the neural network from the files 
	auto net = readNet(file_path + "frozen_inference_graph.pb",
		file_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt", "TensorFlow");


	// Open up the webcam 
	VideoCapture cap(0);

	// Run on either CPU or GPU 
	// with CPU : 30FPS 
	// with GPU : 100 FPS 
	 net.setPreferableBackend(DNN_BACKEND_CUDA); 
	 net.setPreferableTarget(DNN_TARGET_CUDA); 
   
   //...
}
```
### Video Demo 
- [视频演示](https://github.com/Think-Big-Do-Small/CudaOpenCVObjectDetectionDemo/blob/457a2b0a9fad9bbbdfec5ec35f693a8794c1d641/Output.avi)

### About Me 
- Computer Science, Master, Shenzhen University
- I am a software engineer 
- I am familar with computer languages, like c++,java,python,c,matlab,html,css,jquery
- I am familar with databases such as mysql, postgresql
- I am familar with flask, apache tomcat
- I am familar with libraries opencv, caffe, keras, tensorflow, openvino
- I am familar with gpu libraries like cuda, cudnn
- I am recently doing some image segmentation projects with c++, python and cuda <br> background matting etc. <br> 

### About Software Development Experience
- RabbitRun(smart file packaging with high speed and efficiency)  <br> 
visit site: [兔子快跑](http://www.aizaozhidao.vip/tuzikuaipao) 

- AI早知道(ai related projects for demostration) <br> 
visit site: [AI早知道](http://www.aizaozhidao.vip) 

