//  main.cpp
//  K-Means
//  Created by Sathvik Birudavolu on 12/12/15.
//  Copyright © 2015 Sathvik Birudavolu. All rights reserved.

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
class Vector;       //TODO change (count%2==0)
class NDimsKmeans;
void cvDisplayVectors(double** means, Vector* vectors, int K ,int numVectors);
Vector* build_uniform_random_point(int dims);
Vector* generate_clusters(int dims, int K, int vector_count);


class Vector{
private:
    double *vectors = NULL;          // Actual point data
    int cluster_num = INT16_MAX;     // The cluster this point belongs to
    int dims = 0;                    // Number of dimensions
    
public:
    double* getVectors() {
        return vectors;
    }
    
    double getVector(int index) {
        assert(index<dims);
        assert(vectors);
        return vectors[index];
    }
    
    void setVectors(double *vectors) {
        if(!this->vectors) {
            vectors = (double*)malloc(sizeof(double)*dims);
        }
        memcpy(this->vectors, vectors, sizeof(double)*dims);
    }
    
    void setVector (int index, double point) {
        assert(index<dims);
        assert(vectors);
        this->vectors[index] = point;
    }
    
    void setDim(int dims) {
        assert(dims>0);
        this->dims = dims;
        
        if(!this->vectors) {
            vectors = (double*)malloc(sizeof(double)*dims);
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
    int numVectors = 0;
    Vector *vectors;
    
public:
    NDimKMeans(int dims, int K, int numVectors, Vector *vectors) {
        this->dims = dims;
        this->K = K;
        this->numVectors = numVectors;
        this->vectors = vectors;
    }
    
public:
    double** getClusters() {
        int count = 0;
        Vector *mean_v = new Vector[K];
        
        bool isFinished = false;
        
        double **means = new double *[K];
        for (int i = 0; i < K; i++) { means[i] = new double[dims]; }
        
        double **means_prev = new double *[K];
        for (int i = 0; i < K; i++) means_prev[i] = new double[dims];
        
        
        for(int i=0; i<K; i++){
            mean_v[i].setDim(dims);
            mean_v[i].setVectors(vectors[rand()%numVectors].getVectors());
            mean_v[i].setClusterNum(i);
        }
        
        while (!isFinished) {
            
            
            for (int i = 0; i < numVectors; i++) {   //Go through All points
                
                
                double *test_p = vectors[i].getVectors();
                
                int min_index = 0;
                double min = INT16_MAX;
                
                for (int j = 0; j < K; j++) {
                    
                    double dist = 0;
                    for (int l = 0; l < dims; l++) { dist += pow(test_p[l] - mean_v[j].getVector(l), 2); }
                    dist = sqrt(dist);

                    if (dist < min) {
                        min = dist;
                        min_index = j;
                    }
                }
                vectors[i].setClusterNum(min_index);
            }
            
            if(count%2==0){
                
                for(int i=0; i<K; i++)
                    for(int j=0; j<dims; j++)
                        means_prev[i][j] = 0;
                
                double *d = new double[K];
                
                for (int l = 0; l < numVectors; l++) {
                    for (int j = 0; j < dims; j++){
                        means_prev[vectors[l].getClusterNum()][j] += vectors[l].getVector(j);
                        if (j == 0)d[vectors[l].getClusterNum()]++;
                    }
                }
                
            
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < dims; k++) {
                        means_prev[l][k] /= d[l];
                        mean_v[l].setVector(k, means_prev[l][k]);
                        
                    }
                
                
            }else{
                
                for(int i=0; i<K; i++)
                    for(int j=0; j<dims; j++)
                        means[i][j] = 0;
                
                
                double *d = new double[K];
                for (int j = 0; j < dims; j++) {
                    for (int l = 0; l < numVectors; l++) {
                        means[vectors[l].getClusterNum()][j] += vectors[l].getVector(j);
                        if (j == 0) d[vectors[l].getClusterNum()]++;
                    }
                }
                
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < dims; k++) {
                        means[l][k] /= d[l];
                        mean_v[l].setVector(k, means[l][k]);
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
Vector* build_uniform_random_point(int dims) {
    Vector* vec = new Vector();
    vec->setDim(dims);
    
    for(int i=0;i<dims;i++)
        vec->setVector(i, rand() % 500); //changed it from 100 to 600.. because clusters seemed too close to each other
    
    return vec;
}

// Function to generate point_count points with dims dimensions
// Groups them into K clusters
Vector* generate_clusters(int dims, int K, int vector_count) {
    Vector* vectors = new Vector[vector_count];
    
    for(int i=0;i<vector_count;i++)
        vectors[i].setDim(dims);
    
    
    const unsigned int num_per_cluster = vector_count / K;
    unsigned int count = 0;
    unsigned int cluster_count = 0;
    
    // The mean around which a cluster is generated
    float std_dev = 20.0f;
    Vector* mean = build_uniform_random_point(dims);

    
    default_random_engine gen;
    for(int i=0;i<vector_count;i++) {
        
        for(int j=0;j<dims;j++) {
    
            normal_distribution<> n_dist(mean->getVector(j), std_dev);
            vectors[i].setVector(j, n_dist(gen));
            vectors[i].setClusterNum(cluster_count);
        }
        
        if(i%num_per_cluster == 0) {
            // Build a new cluster
            mean = build_uniform_random_point(dims);
            cluster_count++;
        }
        
        count++;
    }
    return vectors;
}

int main() {
    
    int dims = 2;            // 2D points
    int K = 3;               // 3 clusters
    int numVectors = 150;    // A total of 150 points
    
    Vector* vectors = generate_clusters(dims, K, numVectors);
    
    for(int i=0; i<numVectors; i++){
        cout << vectors[i].getVector(0) << "     " << vectors[i].getVector(1) << endl;
    }
    
    // I got these points online i put in only 5 because it would be easier to track where my algorithm was going wrong
    // and the correct means for the points was A(0.7,1) and B(2.5 , 4.5)

    //  points[0].setPoints(new double[2]{1,1});
    //  points[1].setPoints(new double[2]{1,0});
    //  points[2].setPoints(new double[2]{4,4});
    //  points[3].setPoints(new double[2]{2,4});
    //  points[4].setPoints(new double[2]{3,5});

    printf("Got here\n");

    NDimKMeans kmeans(dims, K, numVectors, vectors); // Each point in test_p has a cluster
    
    double** means = kmeans.getClusters();
    
    for(int i=0; i < K; i++){
        cout << means[i][0] << " "<< means[i][1] << " " <<endl;
    }
    
    cvDisplayVectors(means, vectors ,K , numVectors);

    return 0;
}

void cvDisplayVectors(double** means, Vector* vectors, int K, int numVectors){
    
    namedWindow("A", WINDOW_FULLSCREEN);
    Mat img = imread("/Users/BSathvik/Downloads/White_Canvas.jpg", CV_LOAD_IMAGE_COLOR);
    
    Scalar mean_colors(0,0,0);
    
    Scalar *colors_clusters = new Scalar[K];
    
    for(int i=0; i<K; i++)
        colors_clusters[i] = *new Scalar(rand()%255,rand()%255,rand()%255);
    
    for(int i=0 ; i<numVectors; i++)
        circle(img, *new Point(vectors[i].getVector(0),vectors[i].getVector(1)), 3, colors_clusters[vectors[i].getClusterNum()] , 3);
    
    for(int i=0; i<K; i++)
        circle(img, *new Point(means[i][0],means[i][1]), 4 , mean_colors,4);
    
    imshow("A", img);
    
    cvWaitKey(0);
    
}
