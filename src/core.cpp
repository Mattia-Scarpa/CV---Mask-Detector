#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <formatter.h>


using namespace cv;

using namespace dnn;

using namespace std;


// Parameters initialization

float objectnessThreshold = 0.5f; // Objectness threshold
float confThreshold = 0.5f; // Confidence threshold
float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;



// Get the names of the output layers
auto getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}



// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    Scalar Color = Scalar(0,!classId*255,classId*255);
    rectangle(frame, Point(left, top), Point(right, bottom), Color, 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Color,1);
}



// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}


void doYOLOonVideo(string path, String modelConfiguration, String modelWeights, int ver) {

    Mat frame;
    stringstream ss;
    ss << ver;
    string version = ss.str();

    cout << "Processing video... " << path << endl;
    cv::VideoCapture capture(path);
    if (!capture.isOpened()) {
        cerr << "[ERROR] Unable to connect to camera" << endl;
    }

    // Load the network

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    // Create a VideoWriter object
    string OutPutName = path.substr(0, path.length() - 4) + "-maskDetectionOutputv"+ version +".avi";
    cv::VideoWriter maskDetectionOut(OutPutName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    while (capture.read(frame)) {

        if (frame.empty()) {
            std::cout << "[ERROR] Unable to capture frame" << endl;
            break;
        }

        // Creating the 4D blobl from the frame
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        // sets the input to the network
        //YOLOv3
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        // YOLOv3
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        // YOLOv3
        postprocess(frame, outs);

        maskDetectionOut.write(frame);
    }
    maskDetectionOut.release();
    capture.release();
}

void doYOLOonImage(string path, String modelConfiguration, String modelWeights, int ver) {

    cout << "Processing image... " << path << endl;
    Mat img = imread(path);
    stringstream ss;
    ss << ver;
    string version = ss.str();

    if (img.empty()) {
        std::cout << "[ERROR] Unable to capture image " << path << endl;
        return;
    }

    string OutPutName = path.substr(0, path.length() - 4) + "-maskDetectionOutputv" + version + ".jpg";

    // Load the network

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    // Creating the 4D blobl from the frame
    Mat blob;
    blobFromImage(img, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    // sets the input to the network
    //YOLOv4
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    // YOLOv4
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    // YOLOv4
    postprocess(img, outs);

    imwrite(OutPutName, img);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

int main(int argc, char const *argv[]) {

  // Load classes
  string classesFile = "../data/obj.names";
  ifstream ifs(classesFile.c_str());
  string line;
  while (getline(ifs, line)) classes.push_back(line);

  //Image formatting (used only to deal with non .jpg images)
/*
  cout << "Starting image format checking..." << endl;
  formatter f("../data/obj/");
  f.convertImgtoJpg();
  f.writetxt("../data/");*/


  string TEST_PATH = "../tests/";

  // getting images to be processed
  vector<string> imgPaths;
  glob(TEST_PATH+"*.jpg", imgPaths);
  // getting videos to be processed
  vector<string> vidPaths;
  glob(TEST_PATH+"*.mp4", vidPaths);

  cout << "A total of " << imgPaths.size() << " images and " << vidPaths.size() << " videos has been found!" << endl;


  int YOLO_ver;
  cout << "Select the YOLO version (insert 3 or 4), default version YOLOv4... ";
  cin >> YOLO_ver;

  String modelConfiguration;
  String modelWeights;

    if (YOLO_ver != 4) {
      std::cout << "Performing YOLOv3..." << endl;

      modelConfiguration = "../cfg/yolov3-obj.cfg";
      modelWeights = "../cfg/yolov3-obj.weights";
  }
  else {
      std::cout << "Performing YOLOv4..." << endl;

      modelConfiguration = "../cfg/yolov4-obj.cfg";
      modelWeights = "../cfg/yolov4-obj.weights";
  }

  for (size_t i = 0; i < imgPaths.size(); i++) {
      doYOLOonImage(imgPaths[i], modelConfiguration, modelWeights, YOLO_ver);
  }
  for (size_t v = 0; v < vidPaths.size(); v++) {
      doYOLOonVideo(vidPaths[v], modelConfiguration, modelWeights, YOLO_ver);
   }

  cout << "DONE!" << endl;
  return 0;
  }



































//
