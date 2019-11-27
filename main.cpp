#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <streambuf>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <ctime>

namespace fs = std::__fs::filesystem;
using namespace std;

double calculate_distance(vector<double> x, vector<double> y){
    double euclidean = 0;
    for (int i =0; i < x.size(); i++) {
        euclidean += pow((y[i] - x[i]), 2);
    }
    return sqrt(euclidean);
}

vector<vector<double>> choose_initial_centroids(vector<vector<double>> feature_vectors, int k){
    vector<vector<double>> centroid_list;
    std::random_device random_device;
    std::mt19937 engine{random_device()};
    std::uniform_int_distribution<int> dist(0, feature_vectors.size() - 1);
    centroid_list.push_back(feature_vectors[dist(engine)]);
    vector<vector<double>> distance_list(feature_vectors.size());
    vector<double> min_dist_list(feature_vectors.size());
    double sum;
    for(int i=1; i < k; i++){
        #pragma omp parallel for shared(feature_vectors, centroid_list, distance_list) default(none)
        for(int j=0; j<feature_vectors.size(); j++){
            for(const vector<double>& centroid : centroid_list) {
                distance_list[j].push_back(calculate_distance(feature_vectors[j], centroid));
            }
        }
        #pragma omp parallel for shared(distance_list, min_dist_list) default(none)
        for(int j=0; j<distance_list.size(); j++){
            min_dist_list[j] = pow(*min_element(distance_list[j].begin(), distance_list[j].end()), 2);
        }
        sum = accumulate(min_dist_list.begin(), min_dist_list.end(), 0);
        for(double & min : min_dist_list){
            min /= sum;
        }

        sum = accumulate(min_dist_list.begin(), min_dist_list.end(), 0);
        double f = (double)rand() / RAND_MAX;
        double rnd = f * sum;
        for(int i=0; i<min_dist_list.size(); i++) {
            if(rnd < min_dist_list[i]){
                centroid_list.push_back(feature_vectors[i]);
                break;
            }
            rnd -= min_dist_list[i];
        }
    }
    return centroid_list;
}

vector<double> calculate_mean_vector(vector<vector<double>> cluster){
    vector<double> result(cluster[0].size(), 0);
    for(vector<double> item : cluster){
//        #pragma omp parallel for shared(result, item) default(none)
        for(int i=0; i < item.size(); i++){
            result[i] += item[i];
        }
    }
    for(double & i : result){
        i /= cluster.size();
    }
    return result;
}

double WCSS(const vector<vector<double>>& cluster, const vector<double>& centroid){
    vector<double> result;
    for(vector<double> item : cluster){
        result.push_back(pow(calculate_distance(item, centroid), 2));
    }
    return accumulate(result.begin(), result.end(), 0);
}

double BCSS(const vector<vector<vector<double>>>& all_clusters, vector<double> centroid){
    vector<double> result;
    for(vector<vector<double>> cluster : all_clusters){
        result.push_back(cluster.size() * pow(calculate_distance(centroid, calculate_mean_vector(cluster)), 2));
    }
    return accumulate(result.begin(), result.end(), 0);
}

vector<vector<vector<double>>> k_means_clustering(vector<vector<double>> feature_vectors, int k) {
    vector<vector<double>> centroid_list = choose_initial_centroids(feature_vectors, k);
    vector<vector<vector<double>>> result_clusters(k);
    for (int i = 0; i <= 10; i++) {
        for(vector<vector<double>> & cluster : result_clusters){
            cluster.clear();
        }
        for (vector<double> vec : feature_vectors) {
            vector<double> compare_list(k, 0);
            #pragma omp parallel for shared(centroid_list, compare_list, vec) default(none)
            for (int j = 0; j < centroid_list.size(); j++) {
                compare_list[j] = calculate_distance(vec, centroid_list[j]);
            }
//            double min = *min_element(compare_list.begin(), compare_list.end());
            int min_index = min_element(compare_list.begin(), compare_list.end()) - compare_list.begin();
            result_clusters[min_index].push_back(vec);
        }
        #pragma omp parallel for shared(centroid_list, result_clusters) default(none)
        for(int j = 0; j < centroid_list.size(); j++){
            centroid_list[j] = calculate_mean_vector(result_clusters[j]);
        }
    }
    return result_clusters;
}

int count(const string& str, const string& search_for){
    int nPos = str.find(search_for, 0);
    int counted = 0;
    while (nPos != string::npos)
    {
        counted++;
        nPos = str.find(search_for, nPos + 1);
    }
    return counted;
}

string trim(const string& s)
{
    size_t start = s.find_first_not_of(' ');
    return (start == string::npos) ? "" : s.substr(start);
}

struct VectorHasher {
    int operator()(const vector<double> &V) const {
        int hash=0;
        for(int i : V) {
            hash+=i; // Can be anything
        }
        return hash;
    }
};

int main() {
    omp_set_num_threads(4);
    string path = "/Users/patrick/CLionProjects/Parallel_K-Means/Set1";
//    string path = "/Users/patrick/CLionProjects/Parallel_K-Means/691_Data";
    vector<string> filenames;
    for(const auto & iter : fs::directory_iterator(path)) {
        filenames.push_back(iter.path());
    }
    vector<vector<double>> all_feature_vectors(filenames.size());
//    unordered_map<vector<double>, string, VectorHasher> dict;
    double start_feature_extraction = omp_get_wtime();
    fs::directory_iterator iterator = fs::directory_iterator(path);
//    #pragma omp parallel for
//    for(fs::directory_entry x=iterator; x != nullptr; x++){
//
//    }

//    fs::directory_iterator container = fs::directory_iterator(path);
//    #pragma omp parallel for
//    for(auto iter = begin(container); iter != end(container); iter++) {
    #pragma omp parallel for shared(all_feature_vectors, filenames) default(none)
    for(int i=0; i < filenames.size(); i++) {
//        cout << iter->path() << endl;
        string file_path = filenames[i];
        ifstream myfile(file_path);
        string line;
        string file_str;
        int lines_code = 0;
        int line_count = 0;
        vector<double> file_features;
        if (myfile.is_open()) {
            while (getline(myfile, line)) {
                line_count++;
                file_str += line + '\n';
                line = trim(line);
                if(!(line.find("import", 0) == 0 || line.find("//", 0) == 0 || line.find("{", 0) == 0 || line.find("}", 0) == 0 || line == "")){
                    lines_code++;
//need to strip whitespace from lines
                }
//                cout << line << '\n';
            }
            myfile.close();
        }
        double code_percent = (double)lines_code / line_count;
        int bracket_count = count(file_str, "{");
        int comment_count = count(file_str, "//") + count(file_str, "*/");
//        cout << "Bracket Count: " << bracket_count << endl;
//        cout << "Comment Count: " << comment_count << endl;
//        cout << "% Code: " << code_percent << endl;
        file_features.push_back(bracket_count);
        file_features.push_back(comment_count);
        file_features.push_back(code_percent);
//        dict[file_features] = file_path;
        all_feature_vectors[i] = file_features;
        file_features.clear();
//        cout << dict[file_features] << endl;

    }

    double elapsed_feature_extraction = omp_get_wtime() - start_feature_extraction;
    cout << "Feature Extraction Time: " << elapsed_feature_extraction << endl;
    double start_clustering = omp_get_wtime();
    vector<double> mean = calculate_mean_vector(all_feature_vectors);
    double s = WCSS(all_feature_vectors, mean);

    vector<vector<vector<double>>> clusters = k_means_clustering(all_feature_vectors, 7);

//    int cluster_index = 0;
//    for(vector<vector<double>> cluster : clusters){
//        cout << "Cluster " << cluster_index + 1 << " size: " << cluster.size() << endl;
////        cout << "Cluster " << cluster_index + 1 << ": " << endl;
////        for(vector<double> v : cluster){
////            cout << dict[v] << endl;
////        }
//        cout << "WCSS: " << WCSS(cluster, calculate_mean_vector(cluster)) << endl << endl;
//        cluster_index++;
//    }
//    double b = BCSS(clusters, mean);
    double elapsed_clustering = omp_get_wtime() - start_clustering;
//    cout << "BCSS: " << b << endl;
    cout << "        Clustering Time: " << elapsed_clustering << endl;
//    test();
}