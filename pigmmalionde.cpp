// cd /home/kali/Documents/diagFull
// g++ main.cpp -o main -larmadillo -llapack -lblas
//  g++ pigmmalionde.cpp -o pigmmalionde -larmadillo -llapack -lblas -lstdc++fs

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

const double INF_PLUS = std::numeric_limits<double>::infinity();
using namespace std;

const std::string CPUFREQ_PATH = "/sys/devices/system/cpu/cpu4/cpufreq/";

// Function to read the current CPU frequency
unsigned int getCurrentFrequency()
{
  std::ifstream file(CPUFREQ_PATH + "scaling_cur_freq");
  unsigned int frequency = 0;
  if (file.is_open())
  {
    file >> frequency;
    file.close();
  }
  return frequency;
}

// Function to set the CPU frequency
bool setFrequency(unsigned int frequency)
{
  std::ofstream file(CPUFREQ_PATH + "scaling_setspeed");

  if (file.is_open())
  {
    file << frequency;
    file.close();
    std::ofstream file2(CPUFREQ_PATH + "scaling_cur_freq");
    if (file2.is_open())
    {
      file2 << frequency;
      file2.close();
    }
    std::ofstream file3(CPUFREQ_PATH + "scaling_max_freq");
    if (file3.is_open())
    {
      file3 << frequency;
      file3.close();
    }
    return true;
  }
  return false;
}

namespace fs = std::experimental::filesystem;
using namespace std;
using namespace arma;
typedef struct
{
  double w;
  rowvec m;
  rowvec covs;
} component;

typedef struct
{
  rowvec hefts;
  mat means;
  mat covs;
} clustering;
typedef struct
{
  rowvec hefts;
  mat means;
  cube covs;
} clustering_full;
bool hasEnding(std::string const &fullString, std::string const &ending);
clustering incremental_partialGMM(string filename, int number_obs, int dim, int n_gaus, int increment_number, string folder, int try_n, float deadline);
string random(string filename, int number_obs, int dim, int increment_number, string folder);
vector<string> split(vector<string> strings, string str, int max_d);
vector<string> getFiles(string path)
{
  vector<string> files = vector<string>(FILES_N);
  const string end1 = ".mat";
  const string end2 = ".log";
  const string anti_end2 = "_info.log";

  for (const auto &entry : fs::directory_iterator(path))
  {
    std::string fileName = entry.path();
    if (hasEnding(fileName, end1))
    {
      files[0] = fileName;
    }
    else if (hasEnding(fileName, end2) && !hasEnding(fileName, anti_end2))
    {
      files[1] = fileName;
    }
  }
  return files;
}
vector<int> read_parameters(string param_file)
{
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
    if (line.find(s1) != std::string::npos)
    {
      params[0] = stoi(line.substr(s1.length(), line.length()));
      i++;
    }
    if (line.find(s2) != std::string::npos)
    {
      params[1] = stoi(line.substr(s2.length(), line.length()));
      i++;
    }
    if (line.find(s3) != std::string::npos)
    {
      params[2] = stoi(line.substr(s3.length(), line.length()));
      i++;
    }
    if (i == 3)
    {
      break;
    }
  }
  params_f.close();
  return params;
}
bool hasEnding(std::string const &fullString, std::string const &ending)
{
  if (fullString.length() >= ending.length())
  {
    // cout << fullString.length() - ending.length() << ending <<endl;
    return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
  }
  else
  {
    return false;
  }
}
int main(int argc, char const *argv[])
{
  string fold = argv[1];

  string folder = "./data/";
  //  int data_id=atoi(argv[2]);
  //  folder=folder+to_string(data_id)+"/";
  vector<string> files;
  files = getFiles(folder);
  vector<int> params = vector<int>();

  params = read_parameters(files[1]);

  string filename = files[0];
  int number_obs = params[1];

  int dim = params[2];
  int n_gaus = params[0];
  int increment_number = atoi(argv[1]);
  int size_increments = number_obs / increment_number;
   // set the deadline
  double deadline = 560000; 
      int t=0;
    system("sync && echo 3 >/proc/sys/vm/drop_caches");
    incremental_partialGMM(filename, number_obs, dim, n_gaus, increment_number, folder, t, deadline);
  

  return 0;
}

bool empty_s(string s)
{
  cout << s.length() << " " << int(s[0]) << endl;
  for (int i = 0; i < s.length(); i++)
  {
    if ((int(s[i]) != 13) || (s[i] != ' '))
      return false;
  }

  return true;
}

std::vector<std::string> tokenizer(const std::string &p_pcstStr, char delim)
{
  std::vector<std::string> tokens;
  std::stringstream mySstream(p_pcstStr);
  std::string temp;

  while (getline(mySstream, temp, delim))
  {
    tokens.push_back(temp);
  }

  return tokens;
}

vector<string> split(vector<string> strings, string str, int max_d)
{
  strings = vector<string>(max_d);
  int currIndex = 0, i = 0;
  int startIndex = 0, endIndex = 0;
  const char seperator = ' ';
  while ((i <= str.length() - 1) && currIndex < max_d)
  {
    string subStr = "";
    while (str[i] == seperator)
    {
      i++;
    }

    startIndex = i;
    while (str[i] != seperator && i != str.length())
    {
      i++;
    }
    endIndex = i;
    subStr.append(str, startIndex, endIndex - startIndex);
    strings[currIndex] = subStr;
    currIndex += 1;
  }
  return strings;
}
rowvec init_vec(int nb)
{
  rowvec y(nb, fill::ones);
  for (int i = 0; i < nb; i++)
    y(i) = i;
  return y;
}
static void pr_times(suseconds_t real, struct tms *tmsstart, struct tms *tmsend, string folder, int inc, int tries, int k)
{
  static long clktck = 0;
  if (clktck == 0) /* fetch clock ticks per second first time */
    if ((clktck = sysconf(_SC_CLK_TCK)) < 0)
      cout << "sysconf error" << endl;
  real = real / (double)clktck;
  suseconds_t user = tmsend->tms_utime - tmsstart->tms_utime;
  user = user / (double)clktck;
  suseconds_t sys = tmsend->tms_stime - tmsstart->tms_stime;
  sys = sys / (double)clktck;

  string filename = folder + "/time/try" + to_string(tries) + "/" + to_string(inc) + "/";
  fs::create_directories(filename);
  // k numero d inc
  filename = filename + "time_" + to_string(k) + ".csv";

  vector<long int> vm0 = {real, user, sys};
  rowvec m0(vm0.size());
  m0 = conv_to<rowvec>::from(vm0);
  m0.save(filename, csv_ascii);
}



string randomv(string filename, int number_obs, int dim, int increment_number, string folder, int try_n)
{
  int kmax = 62;

  int size_increments = number_obs / kmax;
  size_t rows_number = static_cast<size_t>(number_obs / increment_number);
  cout << rows_number;
  int size_block = size_increments / increment_number;

  // create a shuffle vector for assigning read observations to a file
  rowvec indexes = init_vec(increment_number);
  indexes = shuffle(indexes);

 
  cube blocks(size_block, dim, increment_number, fill::zeros);

  ifstream file;
  string line;
  file.open(filename);
  double value;
  int o = 0, ko = 0;


  string inc_folder = folder + "/try" + to_string(try_n) + "/" + to_string(increment_number) + "/";
  fs::create_directories(inc_folder);
  cout << "stating randomization..." << endl;
  for (int i = 0; i < kmax; i++)
  { // for each increments
    for (int obs = 0; obs < size_block; obs++)
    { // for io block
      for (int k = 0; k < increment_number; k++)
      { // for each obs in an io block
        getline(file, line);
        int idx = indexes(k);
        vector<string> strings;
        strings = split(strings, line, dim);
        // std::cout << "/* message */" << std::endl;

        for (int d = 0; d < dim; d++)
        {

          value = std::stod(strings[d]);

          // Assign the converted value to the blocks vector
          blocks(obs, d, idx) = value;
          //    if (obs==0 && i==0 && k==0) { cout << value <<"   ";}
        }
      }
      //  if (obs==0)   cout <<"loool"<<endl;

      indexes = shuffle(indexes);
    }
    std::vector<std::vector<double>> data;

    // Populate data vector from blocks

    // right in files
    for (int k = 0; k < increment_number; k++)
    { // for each obs in an io block
      string inc_file = inc_folder + to_string(k) + ".rand";
      mat data = reshape(blocks.slice(k), size_block, dim);

      std::ofstream outfile(inc_file, std::ios::binary | std::ios::app);
      if (outfile.is_open())
      {
        // Write the dimensions of the data vector

        size_t rows = data.n_rows * kmax;
        size_t cols = data.n_cols;
        //   std::cout << "cols" << cols<< std::endl;
        //       std::cout << "rows" << rows<< std::endl;

        if (o < increment_number)
        {
          outfile.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
          outfile.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
          o++;
        }

        //    Write the actual data of the data vector
        // Write the actual data of the data vector

        for (size_t i = 0; i < data.n_rows; ++i)
        {
          for (size_t j = 0; j < cols; ++j)
          {
            double value = data(i, j);
            //   if (ko==0){cout <<value<<" ** ";}

            outfile.write(reinterpret_cast<const char *>(&value), sizeof(value));
          }
          ko++;
        }

        outfile.close();
        //  std::cout << "Data written to binary file successfully." << std::endl;
      }
      else
      {
        std::cerr << "Error: Unable to open file " << inc_file << " for writing." << std::endl;
        return "sds";
      }
    }
  }
  file.close();
  cout << "randomization done." << endl;
  return inc_folder;
}

rowvec inv_diag(rowvec s)
{
  int n_elem = s.n_elem;
  rowvec inv = rowvec(n_elem);
  for (int i = 0; i < n_elem; i++)
  {
    inv(i) = 1 / s(i);
  }
  return inv;
}
double trace_diag(rowvec s)
{
  int n_elem = s.n_elem;
  double t = 0;
  for (int i = 0; i < n_elem; i++)
  {
    t = t + s(i);
  }
  return t;
}
double det_diag(rowvec s)
{
  int n_elem = s.n_elem;
  double t = 1;
  for (int i = 0; i < n_elem; i++)
  {
    t = t * s(i);
  }
  return t;
}

double KL_diag(rowvec m0, rowvec s0, rowvec m1, rowvec s1, bool verbose)
{
  int N = m0.n_elem;
  rowvec iS1 = inv_diag(s1);
  rowvec diff = m1 - m0;
  rowvec multip = iS1 % s0;

  double tr_term = trace_diag(multip);
  double det_term = log(det_diag(s1) / det_diag(s0));
  double quad_term = trace_diag(diff % diff % iS1);
  double kl = .5 * (tr_term + det_term + quad_term - N);
  if (verbose)
  {
    std::cout << "tr_term: " << tr_term << std::endl;
    std::cout << "det_term: " << det_term << std::endl;
    std::cout << "quad_term: " << quad_term << std::endl;
    std::cout << "KL: " << kl << std::endl;
  }
  return kl;
}
int assert_params(mat means0, mat covs0)
{
  assert(means0.n_rows == covs0.n_rows);
  return means0.n_rows;
}
int assert_clusters(mat means0, mat means1)
{
  assert(means0.n_rows == means1.n_rows);
  return means0.n_rows;
}
mat KL_dig_matrix(mat means0, mat covs0, mat means1, mat covs1)
{
  int num_comp0 = assert_params(means0, covs0);
  int num_comp1 = assert_params(means1, covs1);
  mat KL_mat(num_comp0, num_comp1, fill::zeros);
  for (int i = 0; i < num_comp0; i++)
  {
    rowvec m0 = means0.row(i);
    rowvec s0 = covs0.row(i);
    for (int j = 0; j < num_comp1; j++)
    {
      rowvec m1 = means1.row(j);
      rowvec s1 = covs1.row(j);
      KL_mat.at(i, j) = KL_diag(m0, s0, m1, s1, false);
    }
  }
  return KL_mat;
}

component merge_two_diag(int n0, double w0, rowvec m0, rowvec s0, int n1, double w1, rowvec m1, rowvec s1)
{
  rowvec new_mean = (n0 * w0 * m0 + n1 * w1 * m1) / (n0 * w0 + n1 * w1);
  double new_weight = (n0 * w0 + n1 * w1) / (n0 + n1);
  rowvec new_cov = (n0 * w0 * s0 + n1 * w1 * s1) / (n0 * w0 + n1 * w1);
  component c;
  c.covs = new_cov;
  c.m = new_mean;
  c.w = new_weight;
  return c;
}
component merge_two_diag_nono(int n0, double w0, rowvec m0, rowvec s0, int n1, double w1, rowvec m1, rowvec s1)
{
  rowvec new_mean = (n0 * w0 * m0 + n1 * w1 * m1) / (n0 * w0 + n1 * w1);
  double new_weight = (n0 * w0 + n1 * w1) / (n0 + n1);
  double w20 = (n0 * w0) / (n0 * w0 + n1 * w1);
  double w21 = (n1 * w1) / (n0 * w0 + n1 * w1);
  w20 = w20 * w20;
  w21 = w21 * w21;
  rowvec new_cov = w20 * s0 + w21 * s1;
  component c;
  c.covs = new_cov;
  std::cout << new_cov.n_rows << " : " << new_cov.n_cols << std::endl;
  c.m = new_mean;
  std::cout << new_mean.n_rows << " : " << new_mean.n_cols << std::endl;
  c.w = new_weight;
  return c;
}

clustering merge_clusters_diag(int n0, int n1, rowvec hefts0, rowvec hefts1, mat means0, mat covs0, mat means1, mat covs1)
{
  clustering clust;
  rowvec hefts(hefts0.n_cols);
  mat means(means0.n_rows, means0.n_cols);
  mat covs(covs0.n_rows, covs0.n_cols);
  int cmp = assert_clusters(means0, means1);
  mat KL = KL_dig_matrix(means0, covs0, means1, covs1);
  rowvec new_hefts;
  mat new_means;
  mat new_covs;
  component c;
  for (int k = 0; k < cmp; k++)
  {
    uword min_index = KL.index_min();
    uvec ii_min = ind2sub(size(KL), min_index);
    int i = ii_min[0];
    int j = ii_min[1];
    c = merge_two_diag(n0, hefts0[i], means0.row(i), covs0.row(i), n1, hefts1[j], means1.row(j), covs1.row(j));
    // clustering.push_back(c);
    hefts[k] = c.w;
    means.row(k) = c.m;
    covs.row(k) = c.covs;

    KL.shed_row(i);
    KL.shed_col(j);
    // c1
    hefts0.shed_col(i);
    means0.shed_row(i);
    covs0.shed_row(i);
    // c2
    hefts1.shed_col(j);
    means1.shed_row(j);
    covs1.shed_row(j);
  }
  clust.covs = covs;
  clust.means = means;
  clust.hefts = hefts;
  return clust;
}

clustering incremental_partialGMM(string filename, int number_obs, int dim, int n_gaus, int increment_number, string folder, int try_n, float deadline)
{
  struct tms tmsstart, tmsend;
  // std::string filenamecsv = "/home/meriem/Desktop/code/data.csv";
  std::string filenamecsv = folder + "/data_DYNN" + to_string(try_n) + ".csv";
  std::ofstream filecsv;
  std::string newLine;

  suseconds_t start, endd;
  int incs_needed;
  int k_max;
  int status;
  float time_chunk, cycle;
  float t_shuffle, t_load, t_inc, slack_time = 0;
  float left = deadline;
  // cout << "Number of increment: " << increment_number << endl;
  unsigned int currentFreq = getCurrentFrequency();
  // std::cout << "Current CPU Frequency: " << currentFreq << " kHz" << std::endl;

  // Set the CPU frequency to a new value (e.g., 2000 MHz) for shuffling
  double newFreqd;
  unsigned int newFreq = 2000000; // 2000 MHz
  bool success = setFrequency(newFreq);
  if (success)
  {
    std::cout << "CPU Frequency set to: " << newFreq << " kHz" << std::endl;
  }
  else
  {
    std::cout << "Failed to set CPU Frequency." << std::endl;
  }
  // Shuffling

  if ((start = times(&tmsstart)) == -1)
    cout << "times error" << endl;

  auto debut = std::chrono::high_resolution_clock::now();

  string inc_folder = folder + "/try" + to_string(try_n) + "/" + to_string(increment_number) + "/";
  randomv(filename, number_obs, dim, increment_number, folder, try_n); // folder+"/"+to_string(increment_number)+"/";

  auto shuffle = std::chrono::high_resolution_clock::now();

  t_shuffle = std::chrono::duration_cast<std::chrono::milliseconds>(shuffle - debut).count();

  if ((endd = times(&tmsend)) == -1)
    cout << "times error" << endl;
  else
    pr_times(endd - start, &tmsstart, &tmsend, folder, increment_number, try_n, 888);
  currentFreq = getCurrentFrequency();

   left=deadline-t_shuffle;

  // writing into the file. aftershuffling.
  filecsv.open(filenamecsv, std::ios::app);
  if (!filecsv.is_open())
  {
    std::cerr << "Error opening the file." << std::endl;
    cout << "cant open it";
  }
  newLine = std::to_string(currentFreq) + "," + std::to_string(t_shuffle) + "," + std::to_string(-1) + "," + std::to_string(left) + "," + std::to_string(0);

  filecsv << newLine << std::endl;
  filecsv.close();

  int size_increments = number_obs / increment_number;
  int kmax = increment_number;

  int n0, n1;
  n0 = 0;
  clustering clust, new_clust;
  float t_inc_worse = 0;
//set to max 
  newFreq = 2000000;
  for (int k = 0; k < kmax; k++)
  {
    cout << "Increment: " << k << endl;
    // read increment file
    string inc_file = inc_folder + to_string(k) + ".rand";
    //  cout <<inc_file<<"blablabla \n \n";
    ifstream file;
    string line;
    // set to minimum while reading

    success = setFrequency(200000);
    if (success)
    {
      std::cout << "loading CPU Frequency set to: 200000" << " kHz" << std::endl;
    }
    else
    {
      std::cout << "Failed to set CPU Frequency." << std::endl;
    }
    currentFreq = getCurrentFrequency();

    std::cout << "Current before loading CPU Frequency: " << currentFreq << " kHz" << std::endl;
    file.open(inc_file);

    int i = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // start of loading
    start_time = std::chrono::high_resolution_clock::now();

    arma::mat inc;
    // Open the input binary file
    std::ifstream infile(inc_file, std::ios::binary);
    if (infile.is_open())
    {
      // Read the dimensions of the inc
      size_t rows, cols;
      if (!infile.read(reinterpret_cast<char *>(&rows), sizeof(rows)) ||
          !infile.read(reinterpret_cast<char *>(&cols), sizeof(cols)))
      {
        std::cerr << "Error: Unable to read dimensions from file " << inc_file << std::endl;
        //  return 1;
      }
     // cout << "number of rows , cols" << rows << " " << cols << endl;
      // Create an Armadillo inc with the read dimensions

      inc.set_size(cols, rows);

      // Read the data from the file into the inc
      for (size_t i = 0; i < rows; ++i)
      {
        for (size_t j = 0; j < cols; ++j)
        {
          double value;
          if (!infile.read(reinterpret_cast<char *>(&value), sizeof(value)))
          {
            std::cerr << "Error: Unable to read data from file " << inc_file << std::endl;
            //  return 1;
          }
          //   if(i==0) {cout <<value <<" ";}
          inc(j, i) = value;
        }
      }

      infile.close();

      // Print the inc
     // std::cout << "inc read from file:" << std::endl;
    }
    else
    {
      std::cerr << "Error: Unable to open file " << inc_file << " for reading." << std::endl;
      // return 1;
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    file.close();
    n1 = inc.n_cols;

    // Calculate the elapsed time in milliseconds
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    t_load = elapsed_time;
    // fin de chargement
    // Print the execution time
    std::cout << "Execution time: reading inc " << t_load << " milliseconds" << std::endl;
    left = left - t_load;

    // writing into the file. after reading an increment.
    filecsv.open(filenamecsv, std::ios::app);
    if (!filecsv.is_open())
    {
      std::cerr << "Error opening the file." << std::endl;
      cout << "cant open it";
    }
    newLine = std::to_string(currentFreq) + "," + std::to_string(elapsed_time) + "," + std::to_string(0) + "," + std::to_string(left) + "," + std::to_string(0);
    filecsv << newLine << std::endl;
    filecsv.close();

    //

    if (k == 0)
    { // si c est le premier fixer la frequence au max
      newFreq = 2000000;
    }

    success = setFrequency(newFreq);
    if (success)
    {
      std::cout << "CPU Frequency set to  holding over: " << newFreq << " kHz" << std::endl;
    }
    else
    {
      std::cout << "Failed to set CPU Frequency." << std::endl;
    }
    currentFreq = getCurrentFrequency();
   // std::cout << "CPU Frequency set to  holding over:*********************** " << currentFreq << " kHz" << std::endl;
    // Learn GMM
    gmm_diag model;
    double percent, log_likelihood; // percent of >=0.95
    double nb_iter;
    cout << "start processing the increment..." << endl;

    // ***************debut  de traitement

    start_time = std::chrono::high_resolution_clock::now();
    // recuperer le nombre d iterations nb_iter

    bool status = model.learn(inc, n_gaus, eucl_dist, random_subset, 10, 100, 1e-5, false, percent, log_likelihood, nb_iter);
    // Record the end time
    end_time = std::chrono::high_resolution_clock::now();
    currentFreq = getCurrentFrequency();
    // Calculate the elapsed time in milliseconds
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    //*********** fin de traitemnt
    // Open the file in append mode

    t_inc = elapsed_time;
    left = left - t_inc;
   // cout << " \n the time to process one chunk " << t_inc << "  number of it: " << nb_iter << "\n";
    // writing into the file. after processing an increment.
    filecsv.open(filenamecsv, std::ios::app);
    if (!filecsv.is_open())
    {
      cout << "cant open it";
    }
    newLine = std::to_string(currentFreq) + "," + std::to_string(elapsed_time) + "," + std::to_string(nb_iter) + "," + std::to_string(left) + "," + std::to_string(0);
    filecsv << newLine << std::endl;
    filecsv.close();
    currentFreq = getCurrentFrequency();
   // std::cout << "CPU Frequency set after     holding over: " << currentFreq << " kHz" << std::endl;

    // std::cout << "Line added to the CSV file." << std::endl;

    // // Print the execution time
    // std::cout << "Execution time: " << elapsed_time << " milliseconds" << std::endl;
    // cout << "Numb3rs of iterations is :" << nb_iter << "\n";

    // cout << "elapsed time for one iteration is " << elapsed_time / nb_iter;
    cout << "\n";

    new_clust.hefts = model.hefts;
    new_clust.means = model.means.t();
    new_clust.covs = model.dcovs.t();

    if (k > 0)
    {
      clust = merge_clusters_diag(n0, n1, clust.hefts, new_clust.hefts, clust.means, clust.covs, new_clust.means, new_clust.covs);
    }
    else
    {
      clust = new_clust;
    }

    // save things--------------------------------------------------------------
    if ((endd = times(&tmsend)) == -1)
      cout << "times error" << endl;
    else
      pr_times(endd - start, &tmsstart, &tmsend, folder, increment_number, try_n, k);

    string foldername = folder + "/model/try" + to_string(try_n) + "/" + to_string(increment_number) + "/";
    fs::create_directories(foldername);
    string filename_m = foldername + "hefts_" + to_string(k) + ".csv";
    clust.hefts.save(filename_m, csv_ascii);

    filename_m = foldername + "covs_" + to_string(k) + ".csv";
    clust.covs.save(filename_m, csv_ascii);

    filename_m = foldername + "means_" + to_string(k) + ".csv";
    clust.means.save(filename_m, csv_ascii);
    //

    n0 = n0 + n1;
    if (k == 0)
    {
      double diff = clust.hefts.min() / clust.hefts.max();

      int data_needed = size_increments + ((percent) * (number_obs - size_increments) + (1 - diff) * (number_obs - size_increments)) / 2;

      incs_needed = int(round(data_needed / size_increments));
      kmax = incs_needed;

      // L de pigmmalion
      cout << "Increments needed:" << incs_needed << endl;

      // calcul how many more increments can we treat
      int nb_restant = int(left / (t_load + t_inc / nb_iter * 100));



      time_chunk = t_load + t_inc / nb_iter * 100;
      t_inc_worse = t_inc / nb_iter * 100;
      cycle = newFreq * t_inc_worse;
      if (nb_restant <= 0)
      {
        kmax = 1;//if we can not treat more than one increment we stop
     //   cout << kmax << "kmax";
      }
      else
      {
        // calculer le nombre d'increment a ignorer pour respecter le temps d'execution
        if (incs_needed > nb_restant)
        {
        //  cout << "skipping " << incs_needed - nb_restant - 1 << "increments\n";
          kmax = nb_restant + 1; // si il reste 1 la condition d arret est 2 ....
        }
        else
        {
          // slack time
          float added = left - (kmax - 1) * t_load - (kmax - 1) * t_inc_worse;
        //  cout << "added=" << added;
          time_chunk = time_chunk + added / (kmax - 1); //  worst time to estimate for the rest of increment
       //   cout << "\n new chunk time:" << time_chunk;
       //taking the sup value of freq
          newFreqd = (cycle / (time_chunk - t_load)) / pow(10, 5);
          newFreq = ceil(newFreqd) * pow(10, 5);

        }
      }

      k_max = kmax;
    }
    if (k > 0)
    {
      // after treating the first increment
      if (elapsed_time < time_chunk - t_load)
      {
       

        float added = left - (kmax - 1 - k) * t_load - (kmax - 1 - k) * time_chunk;
     //   cout << "added=" << added;
        if (kmax != 1 - k)
          time_chunk = time_chunk + added / (kmax - 1 - k); //  nv temps estime pour chaque increment
        cout << "\n new wcettime:" << time_chunk;
        newFreqd = (cycle / (time_chunk - t_load)) / pow(10, 5);
        newFreq = ceil(newFreqd) * pow(10, 5);

        // cout << "\n cycles " << cycle;
      }
      else
      {
        // if no gain in time we continue with the previous wcet hence the same frequency

        newFreqd = (cycle / (time_chunk - t_load)) / pow(10, 5);
        newFreq = ceil(newFreqd) * pow(10, 5);
      }
    }
  }

  auto finn = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(finn - debut).count();
  filecsv.open(filenamecsv, std::ios::app);
  if (!filecsv.is_open())
  {
    std::cerr << "Error opening the file." << std::endl;
    cout << "cant open it";
  }
  // write results
  newLine = std::to_string(elapsed_time - t_shuffle) + "," + std::to_string(deadline) + "," + std::to_string(left) + "," + std::to_string(incs_needed) + "," + std::to_string(k_max);
  filecsv << newLine << std::endl;

  filecsv.close();

  // delete rand files
  for (int k = 0; k < increment_number; k++)
  { // for each obs in an io block
    string inc_file = inc_folder + to_string(k) + ".rand";
 
    int stat = remove(inc_file.c_str());
  }

  // register the weights, means and covs of the final GMM 
  string foldername = folder + "/model/try" + to_string(try_n) + "/" + to_string(increment_number) + "/";
  fs::create_directories(foldername);
  string filename_m = foldername + "hefts" + to_string(deadline / 1000) + ".csv";
  clust.hefts.save(filename_m, csv_ascii);

  filename_m = foldername + "covs" + to_string(deadline / 1000) + ".csv";
  clust.covs.save(filename_m, csv_ascii);

  filename_m = foldername + "means" + to_string(deadline / 1000) + ".csv";
  clust.means.save(filename_m, csv_ascii);
  return clust;
}
