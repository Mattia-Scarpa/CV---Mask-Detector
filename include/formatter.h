#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class formatter {
public:

// constructors
formatter(std::string path);

void convertImgtoJpg();

void writetxt(std::string cfgpath, float VAL_RATIO = 0.2f);

private:
  std::string imgPath;
  std::vector<std::string> imgsPath;  //images path
};
