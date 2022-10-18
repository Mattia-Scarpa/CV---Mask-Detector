#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <formatter.h>

//--------------------TEMPLATES--------------------

template <typename T>
T remove_at(std::vector<T>&v, typename std::vector<T>::size_type n) {
    T ans = std::move_if_noexcept(v[n]);
    v[n] = std::move_if_noexcept(v.back());
    v.pop_back();
    return ans;
}

//--------------------CONSTRUCTRS--------------------

formatter::formatter(std::string path) {
  imgPath = path;
}

//--------------------FUNCTIONS--------------------

void formatter::convertImgtoJpg() {

  std::vector<std::string> imgsPathtemp;
  std::vector<std::string> format = {"*.jpg","*.jpeg","*.png"};

  for (size_t i = 0; i < format.size(); i++) {
    std::cout << "Looking for " << imgPath+format[i] << std::endl;
    cv::glob(imgPath+format[i], imgsPathtemp);
    for (size_t j = 0; j < imgsPathtemp.size(); j++) {

      std::cout << "Processing image" << imgsPathtemp[j] << std::endl;

      cv::Mat img = cv::imread(imgsPathtemp[j], cv::IMREAD_COLOR);
      cv::imwrite(imgsPathtemp[j].substr(0, imgsPathtemp[j].length()-format[i].length()+1)+".jpg", img);

    }
    imgsPathtemp.clear();
  }
}


void formatter::writetxt(std::string cfgpath, float VAL_RATIO) {

  // Removing old configuration files
  remove((cfgpath+"test.txt").c_str());
  remove((cfgpath+"train.txt").c_str());
  remove((cfgpath+"obj.names").c_str());
  remove((cfgpath+"obj.data").c_str());

  std::cout << "Writing conf datafiles" << std::endl;

  cv::glob(imgPath+"*.jpg",imgsPath);

  int VAL_SIZE = imgsPath.size() * VAL_RATIO; // 20% as default

  std::ofstream valFile;
  valFile.open(cfgpath+"test.txt");
  for (size_t i = 0; i < VAL_SIZE; i++) {
    std::srand(std::time(0));
    int index = rand() % imgsPath.size() + 1;
    std::string pathtemp = remove_at(imgsPath, index);
    valFile << pathtemp.substr(2,pathtemp.length()) << std::endl;
  }
  valFile.close();
  std::ofstream objNamesFile, objDataFile, trainFile;

  // setting remaining images for train set
  trainFile.open(cfgpath+"train.txt");
  for (size_t i = 0; i < imgsPath.size(); i++) {
    trainFile << imgsPath[i].substr(2,imgsPath[i].length()-1) << std::endl;
  }
  trainFile.close();

  //other configuration files
  objNamesFile.open(cfgpath+"obj.names");
  objNamesFile << "Mask" << std::endl;
  objNamesFile << "No Mask" << std::endl;
  objNamesFile.close();

  objDataFile.open(cfgpath+"obj.data");
  objDataFile << "classes = 2" << std::endl;
  objDataFile << "train  = data/train.txt" << std::endl;
  objDataFile << "valid  = data/test.txt" << std::endl;
  objDataFile << "names = data/obj.names" << std::endl;
  objDataFile << "backup = backup/" << std::endl;
  objDataFile.close();
}
