//  main.cpp
//  K-Means
//  Created by Sathvik Birudavolu on 12/12/15.
//  Copyright © 2015 Sathvik Birudavolu. All rights reserved.
//
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <assert.h>
#include <float.h>


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>


using namespace cv;
using namespace std;

// Prototypes
class Vector;       //TODO CUDA
class NDimsKmeans;
void cvDisplayVectors(double** means, Vector* vectors, int K ,int numVectors);
Vector* build_uniform_random_point(int dims);
Vector* generate_clusters(int dims, int K, int vector_count);
void cvDisplayVectors(double** means, Vector* vectors, int K, int numVectors);


class Vector{
private:
    double *vectors = NULL;          // Actual point data
    int cluster_num = -1;     // The cluster this point belongs to
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
        bool isFirst = true;
        Vector *mean_v = new Vector[K];
        
        double **means = new double *[K];
        for (int i = 0; i < K; i++) { means[i] = new double[dims]; }
        
        mean_v = randomKMeansPlusPlusInit();
        
        
        while (true) {
            for(int i=0; i < K; i++){
                cout << means[i][0] << " "<< means[i][1] << " " <<endl;
            }
            
            for (int i = 0; i < numVectors; i++) {   //Go through All points
                
                
                double *test_p = vectors[i].getVectors();
                
                int min_index = 0;
                double min = DBL_MAX;
                
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
    
                
            for(int i=0; i<K; i++)
               for(int j=0; j<dims; j++)
                    means[i][j] = 0;
                
            
            double *d = new double[K];
            for(int i=0; i<K; i++) d[i] = 0;
            
            for (int j = 0; j < dims; j++) {
                for (int l = 0; l < numVectors; l++) {
                    means[vectors[l].getClusterNum()][j] += vectors[l].getVector(j);
                    if (j == 0) d[vectors[l].getClusterNum()]++;
                }
            }

            for (int l = 0; l < K; l++)
                for (int k = 0; k < dims; k++) {
                    means[l][k] /= d[l];
                }
            
            if(!isFirst){
                              int c = 0;
              for (int i = 0; i < K; i++)
                  for (int k = 0; k < dims; k++) {
                      if (means[i][k] == mean_v[i].getVector(k)) { c++; }
              }
              if (c == K * dims) break ;
            }
            isFirst = false;
            
            for (int l = 0; l < K; l++)
               for (int k = 0; k < dims; k++) {
                    mean_v[l].setVector(k, means[l][k]);
               }
        }

        return means;
        
    }
    
private:
    Vector* randomKMeansPlusPlusInit (){
        
        Vector* v = new Vector[K];
        
        int r = rand()%(numVectors-1);
        v[0].setDim(dims);
        v[0].setVectors(vectors[r].getVectors());
        
        for(int k=1; k<K; k++){
            double* prob = new double[numVectors];
            
            for(int i=0; i<numVectors; i++){
                
                double min = *new double;
                for(int j=0; j<k; j++){

                    double dist = *new double;
                    dist = 0;
                    for(int l=0; l<dims; l++) dist += pow(v[j].getVector(l)-vectors[i].getVector(l),2);
                    
                    if(j==0) memcpy(&min, &dist, sizeof(double));
                    else if(j > 0 && dist <= min) memcpy(&min, &dist, sizeof(double));
                }
                prob[i] = *new double;
                memcpy(&prob[i], &min, sizeof(double));
            }
            
            int sum_dist_ar = 0;
            for(int i=0; i<numVectors; i++) sum_dist_ar += prob[i];
            
            for(int i=0; i<numVectors; i++) prob[i] /= sum_dist_ar;
            
            double* cumprob = new double[numVectors];
            
            for(int i=0; i<numVectors; i++){
                double c_prob = 0;
                for(int j=0; j<=i; j++)
                    c_prob += prob[j];
                
                cumprob[i] = *new double;
                memcpy(&cumprob[i], &c_prob, sizeof(double));
            }
            
            double rand_prob = rand()/((double)RAND_MAX);
            int index = 0;
            
            for(int  i=0; i<numVectors; i++){
                if(rand_prob < cumprob[i]){
                    index = i;
                    break;
                }
            }

            v[k].setDim(dims);
            v[k].setVectors(vectors[index].getVectors());
        }
        
        for(int i=0; i< K; i++){
            cout << "                      " << v[i].getVector(0) << "  " << v[i].getVector(1) << endl;
        }
        
        return v;
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
    srand(1222);
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

int main() {
    
    int dims = 2;            // 2D points
    int K = 3;               // 3 clusters
    int numVectors = 150;    // A total of 150 points
    
    Vector* vectors = generate_clusters(dims, K, numVectors);
    //Vector* vectors = new Vector[numVectors];
    
    for(int i=0; i<numVectors; i++){
        //vectors[i].setDim(dims);
        //cout << vectors[i].getVector(0) << "     " << vectors[i].getVector(1) << endl;
    }
    
    // I got these points online i put in only 5 because it would be easier to track where my algorithm was going wrong
    // and the correct means for the points was A(0.7,1) and B(2.5 , 4.5)
    /*
      vectors[0].setVectors(new double[2]{1,1});
      vectors[1].setVectors(new double[2]{1,0});
      vectors[2].setVectors(new double[2]{0,2});
      vectors[3].setVectors(new double[2]{2,4});
      vectors[4].setVectors(new double[2]{3,5});
    */
    printf("Got here\n");

    NDimKMeans kmeans(dims, K, numVectors, vectors); // Each point in test_p has a cluster
    
    double** means = kmeans.getClusters();
    
    
    
    for(int i=0; i < K; i++){
        cout << means[i][0] << " "<< means[i][1] << " " <<endl;
    }
    
    cvDisplayVectors(means, vectors ,K , numVectors);
    //getchar();
    return 0;
}
