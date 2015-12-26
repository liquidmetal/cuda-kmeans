
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
void display_vectors(double** means, Vector* vectors, int K ,int numVectors);
Vector* build_uniform_random_point(int dims);
Vector* generate_clusters(int dims, int K, int vector_count);


class Vector {
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

    void print_vector() {
        if(!this->vectors)
            return;

        for(int i=0;i<dims;i++) {
            printf("%f ", vectors[i]);
        }
        printf("\n");
    }
};


class GpuKmeans {
    // TODO
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
    
private:
    Vector* randomKMeansPlusPlusInit (){
        
        Vector* v = new Vector[K];

        int r = rand()%numVectors;
        Vector rand_vec = vectors[r];

        double sum_d_square = 0;
        
        for(int i=0; i<numVectors; i++)
            for(int j=0; j<dims; j++)
                sum_d_square += pow(vectors[i].getVector(j)-rand_vec.getVector(j), 2);
          
        

        double* prob = new double[numVectors];
        double* dist_ar = new double[numVectors];
        
        for(int k=0; k<K; k++){
            for(int i=0; i<numVectors; i++){
                double dist_square = 0;
                
                for(int j=0; j<dims; j++)
                    dist_square += pow(vectors[i].getVector(j)-rand_vec.getVector(j), 2);
                
                
                double min = dist_square;
                if(i==0) dist_ar[i] = min;
                if(i!=0 && dist_ar[i] > min) dist_ar[i] = min;
                
                prob[i] = dist_ar[i]/sum_d_square;
            }
            
            double* prob_sort = new double [numVectors];
            sort(prob, prob + numVectors);
            
            double* cumProb = new double[numVectors-1];
            
            for(int i=0; i< numVectors-1; i++){
                if(i==0)cumProb[0] = prob_sort[0];
                else cumProb[i] += cumProb[i-1]+prob_sort[i];
            }
            
            double ra = (double)(rand()/(RAND_MAX));
            int I = 0;
            for(int i=0; i<numVectors-1; i++)
                if(ra <= cumProb[i]!=1&&cumProb[i] ){
                    I = i; break;
                }
            v[k] = vectors[I];
        }
        
        return v;
    }
};

// Returns a uniformly random n-dimensional vector
Vector* build_uniform_random_point(int dims) {
    Vector* vec = new Vector();
    vec->setDim(dims);
    
    for(int i=0;i<dims;i++) {
        vec->setVector(i, ((double)rand() / RAND_MAX) * 500);
    }
    
    return vec;
}

// Function to generate point_count points with dims dimensions
// Groups them into K clusters
Vector* generate_clusters(int dims, int K, int vector_count) {
    Vector* vectors = new Vector[vector_count];
    
    for(int i=0;i<vector_count;i++) {
        vectors[i].setDim(dims);
    }
    
    // Ensure the number of vectors to generate is divisible by K 
    assert(vector_count % K == 0);
    const unsigned int num_per_cluster = vector_count / K;
    
    // The mean around which a cluster is generated
    const float std_dev = ((double)rand() / RAND_MAX) * 30.0f;

    // Generate K mean positions
    Vector **means = (Vector**)malloc(sizeof(Vector*)*K);
    for(int i=0;i<K;i++)
        means[i] = build_uniform_random_point(dims);
    
    default_random_engine gen;
    unsigned int current_cluster = -1;
    for(int i=0;i<vector_count;i++) {
        // Build a new cluster
        if(i % num_per_cluster == 0)
            current_cluster++;

        // Fetch the n-dimensional center for the current cluster
        Vector* mean = means[current_cluster];

        // Need to set the cluster's number only once
        vectors[i].setClusterNum(current_cluster);

        for(int j=0;j<dims;j++) {
            normal_distribution<> n_dist(mean->getVector(j), std_dev);
            vectors[i].setVector(j, n_dist(gen));
        }
        
    }
    return vectors;
}

int main() {
    int dims = 2;            // 2D points
    int K = 3;               // 3 clusters
    int num_vectors = 150;    // A total of 150 points
   
    // Generate `num_vectors` vectors with `dims` dimensions. Also, divide `num_vectors`
    // points into `K` clusters. 
    Vector* vectors = generate_clusters(dims, K, num_vectors);
    
    NDimKMeans kmeans(dims, K, num_vectors, vectors); // Each point in test_p has a cluster
   
    // TODO why does kmeans return double** and not vector*
    double** means = kmeans.getClusters();
    
    for(int i=0; i < K; i++){
        cout << means[i][0] << " "<< means[i][1] << " " <<endl;
    }
    
    display_vectors(means, vectors, K , num_vectors);

    return 0;
}

void display_vectors(double** means, Vector* vectors, int K, int numVectors){
    cv::Mat img = cv::Mat(600, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Scalar* colors_clusters = new cv::Scalar[K];
    
    for(int i=0; i<K; i++)
        colors_clusters[i] = cv::Scalar(((double)rand()/RAND_MAX)*192,
                                        ((double)rand()/RAND_MAX)*192,
                                        ((double)rand()/RAND_MAX)*192);
    
    for(int i=0 ; i<numVectors; i++) {
        cv::Point pt = cv::Point(vectors[i].getVector(0), vectors[i].getVector(1));
        cv::circle(img, pt, 3, colors_clusters[vectors[i].getClusterNum()] , 3);
    }
    
    cv::Scalar mean_colors(0, 0, 0);
    for(int i=0; i<K; i++) {
        cv::Point pt = cv::Point(means[i][0], means[i][1]);
        circle(img, pt, 4 , mean_colors, 4);
    }
    
    cv::imshow("output clusters", img);

    // Block until the user presses a key
    cv::waitKey(0);
}
