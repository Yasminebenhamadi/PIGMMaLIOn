//cd /home/kali/Documents/diagFull
//g++ main.cpp -o main -larmadillo -llapack -lblas
// g++ v4.cpp -o main -larmadillo -llapack -lblas -lstdc++fs
// with new split function

#include <stdio.h>
#include <iostream>
#include <vector>
#include <armadillo>
#include <omp.h>
#include <assert.h>
#include <limits>
#include <filesystem>
#include <experimental/filesystem>
#include <cmath>
#include <sys/times.h>
#include <iterator>
#define FILES_N 2
#define PARAMS_N 3

const double  INF_PLUS = std::numeric_limits<double>::infinity();


namespace fs = std::experimental::filesystem;
using namespace std;
using namespace arma;
typedef struct {
    double w;
    rowvec m;
    rowvec covs;
}component;

typedef struct {
    rowvec hefts;
    mat means;
    mat covs;
}clustering;
typedef struct {
    rowvec hefts;
    mat means;
    cube covs;
}clustering_full;
bool hasEnding (std::string const &fullString, std::string const &ending) ;
clustering incremental_partialGMM(string filename, int number_obs, int dim, int n_gaus, int increment_number, string folder, int try_n);
string random(string filename, int number_obs, int dim, int increment_number, string folder);
vector<string> split (vector<string> strings, string str, int max_d);
vector<string>  getFiles(string path){
    vector<string> files = vector<string>(FILES_N);
    const string end1 = ".mat";
    const string end2 = ".log";  const string anti_end2 = "_info.log";

    for (const auto & entry : fs::directory_iterator(path)){
        std::string fileName = entry.path();
        if (hasEnding (fileName, end1)){
            files[0]= fileName;
        }
        else if (hasEnding (fileName, end2) && !hasEnding (fileName, anti_end2)){
            files[1]= fileName;
        }
    }
    return files;
}
vector<int> read_parameters(string param_file){
    vector<int> params = vector<int>(PARAMS_N);
    ifstream params_f;
    string line;
    string s1 = "Number of clusters >> ";
    string s2 = "Number of data points >> ";
    string s3 = "Number of dimensions >> ";
    params_f.open(param_file);
    int i = 0;
    while (getline(params_f, line))
    {
        if (line.find(s1) != std::string::npos){
            params[0] = stoi(line.substr (s1.length(),line.length()));
            i++;
        }
        if (line.find(s2) != std::string::npos){
            params[1] = stoi(line.substr (s2.length(),line.length()));
            i++;
        }
        if (line.find(s3) != std::string::npos){
            params[2] = stoi(line.substr (s3.length(),line.length()));
            i++;
        }
        if (i==3){
            break;
        }
    }
    params_f.close();
    return params;
}
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        //cout << fullString.length() - ending.length() << ending <<endl;
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}
int main(int argc,char const *argv[])
{
 
    
   string folder = "./data/"; //"/media/odroid/BEC5-CC55/paper_tests/"+fold+"/";
  //  int data_id=atoi(argv[2]); 
  //  folder=folder+to_string(data_id)+"/";
    vector<string> files;
    files = getFiles(folder);
    vector<int> params = vector<int>();

    params = read_parameters(files[1]);
  
    string filename= files[0];
    int number_obs = params[1];
   
    int dim = params[2];
    int n_gaus=params[0];
    int increment_number = atoi(argv[1]); // ie the size of dataset / the size of an data
    int size_increments=number_obs/increment_number;
    //incremental_partialGMM(filename, number_obs, dim, n_gaus, increment_number, folder, 888);
   // random(filename, number_obs, dim, increment_number, folder);
   for (int t=0;t<1;t++){
   incremental_partialGMM(filename, number_obs, dim, n_gaus, increment_number, folder, t);
   }
    //random(filename, number_obs, dim, increment_number, folder);


    
            
    return 0;
}

bool empty_s(string s){
    cout << s.length() <<" " << int(s[0]) << endl;
    for(int i=0; i<s.length(); i++){
        if ((int(s[i])!=13) || (s[i]!=' '))
            return false;
    }

    return true;
}

std::vector<std::string> tokenizer( const std::string& p_pcstStr, char delim )  {
        std::vector<std::string> tokens;
        std::stringstream   mySstream( p_pcstStr );
        std::string         temp;

        while( getline( mySstream, temp, delim ) ) {
            tokens.push_back( temp );
        }

        return tokens;
    }
vector<string> split (vector<string> strings, string str, int max_d)
{
    strings =  vector<string>(max_d);
    int currIndex = 0, i = 0;
    int startIndex = 0, endIndex = 0;
    const char seperator = ' ';
    while ((i <= str.length()-1) && currIndex<max_d)
    {
        string subStr = "";
        while(str[i] == seperator){
            i++;
        }

        startIndex = i;
        while(str[i] != seperator && i != str.length()){
            i++;
        }
        endIndex=i;
        subStr.append(str, startIndex, endIndex - startIndex);
        strings[currIndex] = subStr;
        currIndex += 1;
    }
    return strings;
}
rowvec init_vec(int nb){
    rowvec y(nb, fill::ones);
    for(int i=0;i<nb;i++)
        y(i) = i;
    return y;
}
static void pr_times(suseconds_t real, struct tms *tmsstart, struct tms *tmsend, string folder, int inc, int tries,int k)
{
    static long clktck = 0;
    if (clktck == 0)    /* fetch clock ticks per second first time */
        if ((clktck = sysconf(_SC_CLK_TCK)) < 0)
            cout << "sysconf error" << endl;
    real = real / (double) clktck;
    suseconds_t user = tmsend->tms_utime - tmsstart->tms_utime;
    user = user/ (double) clktck;
    suseconds_t sys = tmsend->tms_stime - tmsstart->tms_stime;
    sys = sys / (double) clktck;

    string filename=folder+"/time/try"+to_string(tries)+"/"+to_string(inc)+"/";
    fs::create_directories(filename);
    //k numero d inc
    filename = filename+"time_"+to_string(k)+".csv";

    vector<long int> vm0 = {real,user,sys};
    rowvec m0(vm0.size());
    m0 = conv_to<rowvec>::from(vm0);
    m0.save(filename, csv_ascii);
}
string random(string filename, int number_obs, int dim, int increment_number, string folder,int try_n){
   int size_obs = 18*dim;
   int size_io_block = 3*4096;

   int size_block =floor(size_io_block/size_obs)*2;
  // cout << "size_block " <<size_block << endl;

   int size_increments=size_block*increment_number;

   int kmax= number_obs/size_increments;

   //create a shuffle vector for assigning read observations to a file
   rowvec indexes = init_vec(increment_number);
   indexes = shuffle(indexes);

   //matrices that will be written to files
   vector<vector<string>> blocks;
   for (int i=0;i<increment_number;i++){
        vector<string> bloc(size_block);
        blocks.push_back(bloc);
   }

   std::vector<int> vector1(increment_number, 0);

   ifstream file; string line;
   file.open(filename);

  // cout << "Number of increment: "<< increment_number << ", Size of block: "<< size_block << endl;

   string inc_folder=folder+"/try"+to_string(try_n)+"/"+to_string(increment_number)+"/";
   fs::create_directories(inc_folder);
  // cout << "stating randomization..." << endl;
   for(int i=0;i<kmax;i++){//for each increments
       for(int obs=0;obs<size_block;obs++){ //for io block
           for(int k=0;k<increment_number;k++){ //for each obs in an io block
                getline(file, line);
                int idx = indexes(k);
                blocks[idx][obs]=line;
           }
           indexes = shuffle(indexes);
        }
        //right in files
        for(int k=0;k<increment_number;k++){ //for each obs in an io block
            string inc_file=inc_folder+to_string(k)+".rand";
            vector<string> inc = blocks[k];
            ofstream outfile;
            outfile.open(inc_file, std::ios::app);
            ostream_iterator<string> output_iterator(outfile, "\n");
            copy(inc.begin(), inc.end(), output_iterator);
            outfile.close();
        }
    }
   file.close();
   cout << "randomization done." << endl;
   return inc_folder;
}

// string random(string filename, int number_obs, int dim, int increment_number, string folder,int try_n){
//    int kmax= increment_number*20;

//    int size_increments=number_obs / kmax;

//    int size_block=size_increments/increment_number;

//    //create a shuffle vector for assigning read observations to a file
//    rowvec indexes = init_vec(increment_number);
//    indexes = shuffle(indexes);

//    //matrices that will be written to files
//    vector<vector<string>> blocks;
//    for (int i=0;i<increment_number;i++){
//         vector<string> bloc(size_block);
//         blocks.push_back(bloc);
//    }

//    std::vector<int> vector1(increment_number, 0);

//    ifstream file; string line;
//    file.open(filename);

//    cout << "Number of increment: "<< increment_number << ", Size of block: "<< size_block << endl;

//    string inc_folder=folder+"/try"+to_string(try_n)+"/"+to_string(increment_number)+"/";
//    fs::create_directories(inc_folder);
//    cout << "stating randomization..." << endl;
//    for(int i=0;i<kmax;i++){//for each increments
//        for(int obs=0;obs<size_block;obs++){ //for io block
//            for(int k=0;k<increment_number;k++){ //for each obs in an io block
//                 getline(file, line);
//                 int idx = indexes(k);
//                 blocks[idx][obs]=line;
//            }
//            indexes = shuffle(indexes);
//         }
//         //right in files
//         for(int k=0;k<increment_number;k++){ //for each obs in an io block
//             string inc_file=inc_folder+to_string(k)+".rand";
           
//             vector<string> inc = blocks[k];
//             ofstream outfile;
//             outfile.open(inc_file, std::ios::app);
//             ostream_iterator<string> output_iterator(outfile, "\n");
//             copy(inc.begin(), inc.end(), output_iterator);
//             outfile.close();
//         }
//     }
//    file.close();
//    cout << "randomization done." << endl;
//    return inc_folder;
// }



rowvec inv_diag(rowvec s){
    int n_elem = s.n_elem;
    rowvec inv = rowvec(n_elem);
    for (int i=0;i<n_elem;i++){
        inv(i) = 1/s(i);
    }
    return inv;
}
double trace_diag(rowvec s){
    int n_elem = s.n_elem;
    double t = 0;
    for (int i=0;i<n_elem;i++){
        t = t+s(i);
    }
    return t;
}
double det_diag(rowvec s){
    int n_elem = s.n_elem;
    double t = 1;
    for (int i=0;i<n_elem;i++){
        t = t*s(i);
    }
    return t;
}

double KL_diag(rowvec m0,rowvec s0, rowvec m1, rowvec s1, bool verbose){
    int N = m0.n_elem;
    rowvec iS1 = inv_diag(s1);
    rowvec diff = m1 - m0;
    rowvec multip = iS1 % s0;

    double tr_term   = trace_diag(multip);
    double det_term  = log(det_diag(s1)/det_diag(s0));
    double quad_term = trace_diag(diff%diff%iS1);
    double kl = .5 * (tr_term + det_term + quad_term - N);
    if(verbose){
        std::cout << "tr_term: "<< tr_term << std::endl;
        std::cout << "det_term: "<< det_term << std::endl;
        std::cout << "quad_term: "<< quad_term << std::endl;
        std::cout << "KL: "<< kl << std::endl;
    }
    return kl;
}
int assert_params(mat means0, mat covs0){
    assert (means0.n_rows==covs0.n_rows);
    return means0.n_rows;
}
int assert_clusters(mat means0, mat means1){
    assert (means0.n_rows==means1.n_rows);
    return means0.n_rows;
}
mat KL_dig_matrix(mat means0, mat covs0, mat means1, mat covs1){
    int num_comp0 = assert_params(means0,covs0);
    int num_comp1 = assert_params(means1,covs1);
    mat KL_mat(num_comp0, num_comp1, fill::zeros);
    for(int i=0;i<num_comp0;i++){
        rowvec m0=means0.row(i);
        rowvec s0=covs0.row(i);
        for(int j=0;j<num_comp1;j++){
            rowvec m1=means1.row(j);
            rowvec s1=covs1.row(j);
            KL_mat.at(i,j)=KL_diag(m0,s0,m1,s1, false);
        }
    }
    return KL_mat;
}

component merge_two_diag(int n0, double w0, rowvec m0, rowvec s0,int n1, double w1, rowvec m1, rowvec s1){
  rowvec new_mean=(n0*w0*m0+n1*w1*m1)/(n0*w0+n1*w1);
  double new_weight=(n0*w0+n1*w1)/(n0+n1);
  rowvec new_cov=(n0*w0*s0+n1*w1*s1)/(n0*w0+n1*w1);
  component c;
  c.covs = new_cov;
  c.m = new_mean;
  c.w = new_weight;
  return c;
}
component merge_two_diag_nono(int n0, double w0, rowvec m0, rowvec s0,int n1, double w1, rowvec m1, rowvec s1){
  rowvec new_mean=(n0*w0*m0+n1*w1*m1)/(n0*w0+n1*w1);
  double new_weight=(n0*w0+n1*w1)/(n0+n1);
  double w20 = (n0*w0)/(n0*w0+n1*w1);
  double w21 = (n1*w1)/(n0*w0+n1*w1);
  w20 = w20*w20;
  w21 = w21*w21;
  rowvec new_cov=w20*s0+w21*s1;
  component c;
  c.covs = new_cov;
  std::cout << new_cov.n_rows <<" : "<< new_cov.n_cols << std::endl;
  c.m = new_mean;
  std::cout << new_mean.n_rows <<" : "<< new_mean.n_cols << std::endl;
  c.w = new_weight;
  return c;
}

clustering merge_clusters_diag(int n0, int n1, rowvec hefts0, rowvec hefts1, mat means0, mat covs0, mat means1, mat covs1){
    clustering clust;
    rowvec hefts(hefts0.n_cols);
    mat means(means0.n_rows,means0.n_cols);
    mat covs(covs0.n_rows, covs0.n_cols);
    int cmp = assert_clusters(means0, means1);
    mat KL=KL_dig_matrix(means0,covs0, means1,covs1);
    rowvec new_hefts;
    mat new_means;
    mat new_covs;
    component c;
    for(int k=0;k<cmp;k++){
      uword min_index = KL.index_min();
      uvec ii_min = ind2sub(size(KL), min_index);
      int i = ii_min[0];
      int j = ii_min[1];
      c = merge_two_diag(n0, hefts0[i], means0.row(i), covs0.row(i), n1, hefts1[j], means1.row(j), covs1.row(j));
      //clustering.push_back(c);
      hefts[k]= c.w;
      means.row(k)= c.m;
      covs.row(k) = c.covs;

      KL.shed_row(i);
      KL.shed_col(j);
      //c1
      hefts0.shed_col(i);
      means0.shed_row(i);
      covs0.shed_row(i);
      //c2
      hefts1.shed_col(j);
      means1.shed_row(j);
      covs1.shed_row(j);
  }
  clust.covs=covs;
  clust.means=means;
  clust.hefts=hefts;
  return clust;
}

clustering incremental_partialGMM(string filename, int number_obs, int dim, int n_gaus, int increment_number, string folder, int try_n){
   struct tms  tmsstart, tmsend;
   suseconds_t     start, endd;
   int         status;
   cout << "Number of increment: "<< increment_number << endl;

   //Shuffling
   if ((start = times(&tmsstart)) == -1)
        cout << "times error" << endl;

   string inc_folder=folder+"/try"+to_string(try_n)+"/"+to_string(increment_number)+"/";
   random(filename, number_obs, dim, increment_number, folder,try_n); //folder+"/"+to_string(increment_number)+"/";
   if ((endd = times(&tmsend)) == -1)
            cout << "times error" << endl;
   else
        pr_times(endd-start, &tmsstart,&tmsend, folder, increment_number, try_n,888);



   int size_increments=number_obs / increment_number;
   int kmax=increment_number;


   int n0, n1; n0=0;
   clustering clust, new_clust;


   for(int k=0;k<kmax;k++){
       cout << "\n Treatement de Increment: "<< k+1 << endl;
        //read increment file
        string inc_file=inc_folder+to_string(k)+".rand";
     //   cout <<inc_file<<"blablabla \n \n";
        ifstream file; string line;
        file.open(inc_file);

        mat inc(dim, size_increments,  fill::zeros);
        int i=0;
        while (getline(file, line) && (i<size_increments))
        {
            vector<string> strings;
            strings = split (strings, line, dim);
            for(int d=0; d<dim;d++){
                inc(d,i)=stod(strings[d]);
            }
            i=i+1;
        }
        if(i<size_increments){
            inc = inc.cols(0,i-1);
        }
        file.close();
        n1=inc.n_cols;

      //  cout << "Increment size: "<< n1 << ", dim:" << inc.n_rows <<endl;

        //Learn GMM
        gmm_diag model;
        double percent, log_likelihood,nb_inc; // percent of >=0.95local
        cout << "handing over..." << endl;
        bool status = model.learn(inc, n_gaus,eucl_dist, random_subset, 10, 100, 1e-5, false, percent, log_likelihood,nb_inc);

        new_clust.hefts=model.hefts;
        new_clust.means=model.means.t();
        new_clust.covs=model.dcovs.t();

        if(k>0){
            clust=merge_clusters_diag(n0, n1, clust.hefts, new_clust.hefts , clust.means, clust.covs, new_clust.means, new_clust.covs);
        }
        else{
            clust=new_clust;
        }

        //save things--------------------------------------------------------------
        if ((endd = times(&tmsend)) == -1)
            cout << "times error" << endl;
       else
            pr_times(endd-start, &tmsstart,&tmsend, folder, increment_number, try_n,k);

       string foldername=folder+"/model/try"+to_string(try_n)+"/"+to_string(increment_number)+"/";
       fs::create_directories(foldername);
       string filename_m = foldername+"hefts_"+to_string(k)+".csv";
     clust.hefts.save(filename_m,csv_ascii);

       filename_m = foldername+"covs_"+to_string(k)+".csv";
      clust.covs.save(filename_m,csv_ascii);

       filename_m = foldername+"means_"+to_string(k)+".csv";
      clust.means.save(filename_m,csv_ascii);
        //

        n0=n0+n1;
        if(k==0){
            double diff = clust.hefts.min()/clust.hefts.max();
            cout << "******";
            cout << "\n value of alpha :" <<  percent <<endl;
            cout << "\n value of  beta :" <<  diff <<endl;
            cout << "******";


            int data_needed = size_increments+((percent)*(number_obs-size_increments) + (1-diff)*(number_obs-size_increments))/2;
            int incs_needed;
            incs_needed = int(round(data_needed/size_increments));
          //  cout << "percent:" << percent << endl;
          //  cout << "data_needed:" << data_needed << endl;
           // cout << "Increments needed:" << incs_needed << endl;
            kmax=incs_needed;
            cout << "lNumber of increments treated by PigmmaliOn is  :"<< kmax;

        }
        
   }
   /*int k = try_n;
   if ((endd = times(&tmsend)) == -1)
            cout << "times error" << endl;
   else
        pr_times(endd-start, &tmsstart,&tmsend, folder, increment_number, k);

   /*string foldername=folder+"/model/"+to_string(increment_number)+"/";
   fs::create_directories(foldername);
   string filename_m = foldername+"hefts_"+to_string(k)+".csv";
   clust.hefts.save(filename_m,csv_ascii);

   filename_m = foldername+"covs_"+to_string(k)+".csv";
   clust.covs.save(filename_m,csv_ascii);

   filename_m = foldername+"means_"+to_string(k)+".csv";
   clust.means.save(filename_m,csv_ascii);*/
   
    //  for(int k=1;k<increment_number;k++){ //for each obs in an io block
    //         string inc_file=inc_folder+to_string(k)+".rand";
    //      //   const char* charPtr=
    //       int stat=  remove(inc_file.c_str());
    //       }
   return clust;
}

