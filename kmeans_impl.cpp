//
//  main.cpp
//  K-Means
//
//  Created by Sathvik Birudavolu on 12/12/15.
//  Copyright © 2015 Sathvik Birudavolu. All rights reserved.
//

#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

// Prototypes
class Points;       // TODO Points needs to change to "Point"
class NDimsKmeans;
void cvDisplayVectors(double** means, Points* points, int K ,int numPoints);
Points* build_uniform_random_point(int dims);
Points* generate_clusters(int dims, int K, int point_count);


class Points{
private:
    double *points = NULL;          // Actual point data
    int cluster_num = INT16_MAX;    // The cluster this point belongs to
    int dims = 0;                   // Number of dimensions
    
public:
    double* getPoints() {
        return points;
    }
    
    double getPoint(int index) {
        assert(index<dims);
        assert(points);
        return points[index];
    }
    
    void setPoints(double *points) {
        if(!this->points) {
            points = (double*)malloc(sizeof(double)*dims);
        }
        memcpy(this->points, points, sizeof(double)*dims);
    }
    
    void setPoint (int index, double point) {
        assert(index<dims);
        assert(points);
        this->points[index] = point;
    }
    
    void setDim(int dims) {
        assert(dims>0);
        this->dims = dims;

        if(!this->points) {
            points = (double*)malloc(sizeof(double)*dims);
        }
    }
    
    int getClusterNum() {
        return cluster_num;
    }
    
    void setClusterNum(int num) {
        cluster_num = num;
    }
};


class NDimKMeans {
private:
    int dims = 0;
    int K = 0;
    int numPoints = 0;
    Points *points;
    
public:
    NDimKMeans(int dims, int K, int numPoints, Points *points) {
        this->dims = dims;
        this->K = K;
        this->numPoints = numPoints;
        this->points = points;
    }
    
public:
    double** getClusters() {
        int count = 0;
        Points *mean_p = new Points[K];
        
        bool isFinished = false;
        
        
        double **means = new double *[K];
        for (int i = 0; i < K; i++) { means[i] = new double[dims]; }
        
        double **means_prev = new double *[K];
        for (int i = 0; i < K; i++) means_prev[i] = new double[dims];
        
        mean_p[0].setPoints(new double[2]{100,110});
        mean_p[0].setClusterNum(0);
        mean_p[1].setPoints(new double[2]{250,200});
        mean_p[1].setClusterNum(1);
        mean_p[2].setPoints(new double[2]{300,300});
        mean_p[2].setClusterNum(2);
        
    //loop:
        while (!isFinished) {
            
            
            for (int i = 0; i < numPoints; i++) {   //Go through All points
                
                
                double *test_p = points[i].getPoints();
                
                //cout <<i << " "<< test_p[0] << " " << test_p[1] << endl;
                int min_index = 0;
                double min = INT16_MAX;
                
                for (int j = 0; j < K; j++) {
                    
                    double dist = 0;
                    for (int l = 0; l < dims; l++) { dist += pow(test_p[l] - mean_p[j].getPoint(l), 2); }
                    dist = sqrt(dist);
                    //cout << numPoints << " "<< i << " " << j << " " << dist << endl;
                    
                    //cout << dist << " " << endl;
                    if (dist < min) {
                        min = dist;
                        //cout << min << " ";
                        min_index = j;
                    }
                }
                cout << endl;
                points[i].setClusterNum(min_index);
                cout << test_p[0] <<"             " << test_p[1] << "    "<< points[i].getClusterNum() << endl;
            }
            
            
            
            if(count%2 == 0){
                
                for(int i=0; i<K; i++)
                    for(int j=0; j<dims; j++)
                        means_prev[i][j] = 0;
                
                double *d = new double[K];
                
                for (int l = 0; l < numPoints; l++) {
                    for (int j = 0; j < dims; j++){
                        means_prev[points[l].getClusterNum()][j] += points[l].getPoint(j);
                        
                        if (j == 0)d[points[l].getClusterNum()]++;
                    }
                }
                
                
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < dims; k++) {
                        means_prev[l][k] /= d[l];
                        mean_p[l].setPoint(k, means_prev[l][k]);
                        
                    }
            
                
            }else{
                
                for(int i=0; i<K; i++)
                    for(int j=0; j<dims; j++)
                        means[i][j] = 0;
                
                
                double *d = new double[K];
                for (int j = 0; j < dims; j++) {
                    for (int l = 0; l < numPoints; l++) {
                        means[points[l].getClusterNum()][j] += points[l].getPoint(j);
                        if (j == 0) d[points[l].getClusterNum()]++;
                    }
                }
                
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < dims; k++) {
                        means[l][k] /= d[l];
                        mean_p[l].setPoint(k, means[l][k]);
                    }
                
                int c = 0;
                for (int i = 0; i < K; i++)
                    for (int k = 0; k < dims; k++) {
                        if (means[i][k] == means_prev[i][k]) { c++; }
                    }
                if (c == K * dims) break ;
            }
            
            count++;
        }
        
        
        return means;
        
    }
};

// Returns a uniformly random vector of size dims
Points* build_uniform_random_point(int dims) {
    Points* vec = new Points();
    vec->setDim(dims);

    for(int i=0;i<dims;i++)
        vec->setPoint(i, rand() % 100);

    return vec;
}

// Function to generate point_count points with dims dimensions
// Groups them into K clusters
Points* generate_clusters(int dims, int K, int point_count) {
    Points* points = new Points[point_count];

    for(int i=0;i<point_count;i++)
        points[i].setDim(dims);


    const unsigned int num_per_cluster = point_count / K;
    unsigned int count = 0;
    unsigned int cluster_count = 0;

    // The mean around which a cluster is generated
    float std_dev = 20.0f;
    Points* mean = build_uniform_random_point(dims);

    default_random_engine gen;
    for(int i=0;i<point_count;i++) {
        for(int j=0;j<dims;j++) {
            //points[i].setPoints(new double[2]{abs(n_dist(gen)),abs(n_dist(gen))});
            normal_distribution<> n_dist(mean->getPoint(j), std_dev);
            points[i].setPoint(j, n_dist(gen));
            points[i].setClusterNum(current_cluster);
        }

        if(i%num_per_cluster == 0) {
            // Build a new cluster
            mean = build_uniform_random_point(dims);
            current_cluster++;
        }

        count++;
    }


    /*for(int j=0;j<K;j++){
        normal_distribution<> n_dist(100+100*j,20);
        for(int i=0 ;i<numPoints/K; i++){
            cout <<  "                      " << n_dist(gen) <<endl;
            points[j*(numPoints/K)+i].setPoints(new double[2]{abs(n_dist(gen)),abs(n_dist(gen))});
            count++;
        }
    }*/

}

int main() {
    
    int dims = 2;           // 2D points
    int K = 3;              // 3 clusters
    int numPoints = 150;    // A total of 150 points

    Points* points = generate_clusters(dims, K, numPoints);
    
    // I got these points online i put in only 5 because it would be easier to track where my algorithm was going wrrong
    // and the correct means for the points was A(0.7,1) and B(2.5 , 4.5) and i got the same answers with more precision
   
  //  points[0].setPoints(new double[2]{100,100});
  //  points[1].setPoints(new double[2]{200,200});
  //  points[2].setPoints(new double[2]{400,400});
    //points[3].setPoints(new double[2]{2,4});
    //points[4].setPoints(new double[2]{3,5});
    
    
    printf("Got here\n");
    
    
    NDimKMeans m(dims, K, numPoints, points); // Each point in test_p has a cluster

    double** means = m.getClusters();

    for(int i=0; i < K; i++){
        cout << means[i][0] << " "<< means[i][1] << " " <<endl;
    }
    
    cvDisplayVectors(means,points,K ,numPoints);

    
    return 0;
}

void cvDisplayVectors(double** means, Points* points, int K, int numPoints){
    
    namedWindow("A", WINDOW_FULLSCREEN);
    Mat img = imread("/Users/BSathvik/Downloads/White_Canvas.jpg", CV_LOAD_IMAGE_COLOR);
    
    Scalar mean_colors(0,0,0);
    
    Scalar *colors_clusters = new Scalar[K];
    
    for(int i=0; i<K; i++)
        colors_clusters[i] = *new Scalar(rand()%255,rand()%255,rand()%255);
    
    
    
    for(int i=0 ; i<numPoints; i++){
        circle(img, *new Point(points[i].getPoint(0),points[i].getPoint(1)), 3, colors_clusters[points[i].getClusterNum()] , 3);
    }
    
    for(int i=0; i<K; i++){
        circle(img, *new Point(means[i][0],means[i][1]), 4 , mean_colors,4);
    }
    
    imshow("A", img);
    
    cvWaitKey(0);
    
}

