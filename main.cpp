#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <fstream>
#include <string>

using namespace std;
using namespace cv;

int main(int, char**){

    std::ifstream inputFile("..\\..\\..\\..\\resources\\report1_F\\list_of_points.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error : Impossible to open the file" << std::endl;
        return 1;
    }

    int num;
    while (cout << "Number of points for the Algorithm : " && !(cin >> num)) {
        cin.clear();
        cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        cout << "Invalid.\n";
    }
    if (num < 0){
        cout << "Error : number must be positive\n";
        inputFile.close();
        return (1);
    }

    std::vector<Point2d> points1;
    std::vector<Point2d> points2;

    cout << "Points loaded : (img1.x ; img1.y / img2.x ; img2.y)" << endl;
    for(int i = 0; i < num; i++) {
        Point2d p1;
        Point2d p2;
        for(int j = 0; j < 9; j++){
            string temp;
            inputFile >> temp;
            if(j == 1) p1.x = stoi(temp);
            if(j == 3) p1.y = stoi(temp);
            if(j == 5) p2.x = stoi(temp);
            if(j == 7) p2.y = stoi(temp);
        }
        points1.push_back(p1);
        points2.push_back(p2);
        cout << p1.x << " ; " << p1.y << " / " << p2.x << " ; " << p2.y << endl;
    }

    inputFile.close();


    // Data and extration of Essential Matrix
    InputArray points1_cv(points1);
    InputArray points2_cv(points2);
    cv::Mat K1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat K2 = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat E, mask;
    E = cv::findEssentialMat(points1_cv,points2_cv,K1,cv::RANSAC, 0.999, 1.0, mask);
    int rows = E.rows;
    int cols = E.cols;

    cout << endl << "Essential matrix E : " << endl;
    cout << E << endl << "Dimensions : [" << rows << " ; " << cols << "]"<< endl << endl;
    

    int number_matrix = rows/3;
    cout << "There are " << number_matrix << " essential matrix possibles :" << endl;
    for(int i = 0 ; i < number_matrix; i++){
        //Split E in number_matrix differents matrix 3*3
        cv::Mat E_temp = E.rowRange(3*i, 3*(i+1));
        cout << "Essential matrix E" << i+1 << " : " << endl;
        cout << E_temp << endl;

        //Save them in files named "matrix_E{i+1}.txt" with i the current number
        string name = "matrix_E" + to_string(i+1);
        name += ".txt";
        std::ofstream file("..\\..\\..\\..\\resources\\report1_F\\A2_matrix_E\\"+ name);
        file << E_temp << std::endl;
        file.close();
    }

    return 0;
}
