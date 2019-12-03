#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/range.hpp>
#include <fstream>
#include <vector>
#include <streambuf>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <ctime>
 
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
        #pragma omp parallel for shared(feature_vectors, centroid_list, distance_list, min_dist_list) default(none)
        for(int j=0; j<feature_vectors.size(); j++){
            for(const vector<double>& centroid : centroid_list) {
                distance_list[j].push_back(calculate_distance(feature_vectors[j], centroid));
            }
            min_dist_list[j] = pow(*min_element(distance_list[j].begin(), distance_list[j].end()), 2);
        }
        sum = accumulate(min_dist_list.begin(), min_dist_list.end(), 0);
        #pragma omp parallel for shared(min_dist_list, sum) default(none)
        for(int j=0; j<min_dist_list.size(); j++){
            min_dist_list[j] /= sum;
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
    omp_lock_t lock;
    omp_init_lock(&lock);
    vector<vector<double>> centroid_list = choose_initial_centroids(feature_vectors, k);
    vector<vector<vector<double>>> result_clusters(k);
    for (int i = 0; i <= 10; i++) {
        for(vector<vector<double>> & cluster : result_clusters){
            cluster.clear();
        }
        double total_compare = 0;
        double start_compare = omp_get_wtime();
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
        double start_centroid = omp_get_wtime();
        #pragma omp parallel for shared(centroid_list, result_clusters) default(none)
        for(int j = 0; j < centroid_list.size(); j++){
            // cout << j << endl;
            centroid_list[j] = calculate_mean_vector(result_clusters[j]);
        }
        double end_centroid = omp_get_wtime() - start_centroid;
        // cout << '\t' << "Centroid time for i = " << i << ": " << end_centroid << endl;
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

int main(int argc, char** argv) {
    omp_set_num_threads(atoi(argv[3]));
    int num_files = atoi(argv[1]);
    string dirpath = "/deac/classes/csc726/willpj18/Parallel_K-Means/Test_Data2";
    boost::filesystem::path dir(dirpath);
    vector<string> filenames;
    int file_count = 0;
    if(is_directory(dir)) {
        for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dir), {})){
            if(file_count == num_files){
                break;
            }
            filenames.push_back(entry.path().string());
            file_count++;
        }
    }
    vector<vector<double>> all_feature_vectors(filenames.size());
    double start_feature_extraction = omp_get_wtime();


    #pragma omp parallel for shared(all_feature_vectors, filenames) default(none)
    for(int i=0; i < filenames.size(); i++) {
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
                }
            }
            myfile.close();
        }
        double code_percent = (double)lines_code / line_count;
        int bracket_count = count(file_str, "{");
        int comment_count = count(file_str, "//") + count(file_str, "*/");
        file_features.push_back(bracket_count);
        file_features.push_back(comment_count);
        file_features.push_back(code_percent);
        all_feature_vectors[i] = file_features;
        file_features.clear();
    }
    double elapsed_feature_extraction = omp_get_wtime() - start_feature_extraction;
    std::cout << "Feature Extraction Time: " << elapsed_feature_extraction << endl;


    double start_clustering = omp_get_wtime();
    vector<double> mean = calculate_mean_vector(all_feature_vectors);
    double s = WCSS(all_feature_vectors, mean);
    vector<vector<vector<double>>> clusters = k_means_clustering(all_feature_vectors, atoi(argv[2]));
    double elapsed_clustering = omp_get_wtime() - start_clustering;
    std::cout << "        Clustering Time: " << elapsed_clustering << endl;
    cout << "             Total Time: " << omp_get_wtime() - start_feature_extraction << endl;
}