#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
using namespace std;


const int NUM_OF_FEATURES = 16;

struct FeaturePoint
{
    int numOfYes;
    int numOfNo;
    FeaturePoint()
    {
        numOfNo = 0;
        numOfYes = 0;
    }
    int getTotal() const { return numOfYes + numOfNo; }
};


struct FeatureVector {
    string className;
    vector<string> featureValues;
    FeatureVector() {}
    FeatureVector(int size) : featureValues(vector<string>(size)) {}
};

class FrequencyTable {
private:

    int numOfRepublicans;
    int numOfDemocrats;
    int totalNum;
    unordered_map<string, vector<FeaturePoint>> frequencyTable;
    vector<string> classesName{ "republican", "democrat" };
    void setNumOfRepublicansAndDemocrats(vector<FeatureVector> data);
public:

    FrequencyTable(vector<FeatureVector> dataset);
    unordered_map<string, vector<FeaturePoint>>& getTable() { return frequencyTable; }
    int getNumOfRepublicans() const { return numOfRepublicans; }
    int getNumOfDemocrats() const { return numOfDemocrats; }
    int getAll() const { return totalNum; }
};
//Private function definitions:

void FrequencyTable::setNumOfRepublicansAndDemocrats(vector<FeatureVector> data)
{
    numOfRepublicans = 0;
    numOfDemocrats = 0;
    totalNum = 0;
    for (FeatureVector featureVector : data)
    {
        if (featureVector.className == "republican")
        {
            numOfRepublicans++;
        }
        else if (featureVector.className == "democrat")
        {
            numOfDemocrats++;
        }
    }
    totalNum = numOfDemocrats + numOfRepublicans;
}

//Public function definitions:

FrequencyTable::FrequencyTable(vector<FeatureVector> dataset)
{
    for (string className : classesName)
    {
        vector<FeaturePoint> featurePointVector;
        for (int i = 0; i < NUM_OF_FEATURES; i++)
        {
            featurePointVector.push_back(FeaturePoint());
        }
        frequencyTable[className] = featurePointVector;
    }
    setNumOfRepublicansAndDemocrats(dataset);
}


class NaiveBayesClassifier {
private:
    vector<FeatureVector> dataset;
    vector<string> splitString(string str);
    bool checkIfContainsUnknown(vector<string> vect);
    void readFromFile(string filename);
    double BayesCalculation(double probability, vector<FeaturePoint> FeaturePoints, vector<string> featureValues);
    string findBestProbability(FrequencyTable table, vector<string> featureValues);

public:
    NaiveBayesClassifier(const char* filename) { readFromFile(filename); }
    FrequencyTable train(vector<FeatureVector> trainingData);
    double test(FrequencyTable frequencyTable, vector<FeatureVector> testingData);
    void run();
};
//Private function definitions:

vector<string> NaiveBayesClassifier::splitString(string str)
{
    istringstream ss(str);
    string token;

    vector<string> row;
    while (getline(ss, token, ','))
    {
        row.push_back(token);
    }
    return row;
}

bool NaiveBayesClassifier::checkIfContainsUnknown(vector<string> vect)
{
    for (int i = 0; i < vect.size(); i++)
    {
        if (vect[i] == "?")
        {
            return true;
        }
    }
    return false;
}

void NaiveBayesClassifier::readFromFile(string filename)
{
    ifstream file(filename);

    vector<FeatureVector> fileData;
    if (file.is_open())
    {
        while (true)
        {
            if (file.eof())
            {
                break;
            }
            string line;
            file >> line;
            if (line == "")
            {
                continue;
            }
            vector<string> newLine = splitString(line);
            FeatureVector featureVector;
            featureVector.className = newLine[0];
            featureVector.featureValues = vector<string>(newLine.begin() + 1, newLine.end());
            if (!checkIfContainsUnknown(featureVector.featureValues))
            {
                dataset.push_back(featureVector);
            }
        }
        file.close();
    }
}
double NaiveBayesClassifier::BayesCalculation(double probability, vector<FeaturePoint> FeaturePoints, vector<string> featureValues)
{
    for (int i = 0; i < featureValues.size(); i++)
    {
        string featureValue = featureValues[i];
        FeaturePoint point = FeaturePoints[i];
        if (featureValue == "y")
        {
            probability += log((double)(1 + point.numOfYes) / (point.getTotal() + 2));
        }
        else if (featureValue == "n")
        {
            probability += log((double)(1 + point.numOfNo) / (point.getTotal() + 2));
        }
    }
    return probability;
}
string NaiveBayesClassifier::findBestProbability(FrequencyTable table, vector<string> featureValues)
{
    double bestResult = numeric_limits<double>::lowest();
    string bestClass = "";
    for (auto const& elem : table.getTable())
    {
        double probability = (elem.first == "democrat") ? table.getNumOfDemocrats() :
            table.getNumOfRepublicans();
        probability = (probability + 1) / (table.getAll() + 2);
        double bayes = BayesCalculation(probability, elem.second, featureValues);
        if (bestResult < bayes)
        {
            bestResult = bayes;
            bestClass = elem.first;
        }
    }

    return bestClass;
}
//Public function definitions:

FrequencyTable NaiveBayesClassifier::train(vector<FeatureVector> trainingData)
{
    FrequencyTable table = FrequencyTable(trainingData);
    for (FeatureVector featureVector : trainingData)
    {
        vector<string> featureValues = featureVector.featureValues;
        for (int i = 0; i < featureValues.size(); i++)
        {
            if (featureValues[i] == "y")
            {
                table.getTable()[featureVector.className][i].numOfYes++;
            }
            else if (featureValues[i] == "n")
            {
                table.getTable()[featureVector.className][i].numOfNo++;
            }
        }
    }
    return table;
}

double NaiveBayesClassifier::test(FrequencyTable frequencyTable, vector<FeatureVector> testingData)
{
    double correctCount = 0.0;
    for (FeatureVector featureVector : testingData)
    {
        string guess = findBestProbability(frequencyTable, featureVector.featureValues);
        if (featureVector.className == guess)
        {
            correctCount++;
        }
    }

    return correctCount / testingData.size();
}

void NaiveBayesClassifier::run()
{
    unsigned seed = 0;
    shuffle(dataset.begin(), dataset.end(), default_random_engine(seed));

    int sizeOfSubset = dataset.size() / 10;
    double average = 0.0;
    int tries = 0;
    for (int i = 0; i < sizeOfSubset * 10; i += sizeOfSubset)
    {
        int startIdx = i;
        int endIdx = (i + sizeOfSubset >= dataset.size()) ? dataset.size() - 1 : i + sizeOfSubset;

        vector<FeatureVector> testingData(dataset.begin() + startIdx, dataset.begin() + endIdx);
        vector<FeatureVector> trainingData(dataset.begin(), dataset.begin() + startIdx);

        trainingData.insert(trainingData.end(), dataset.begin() + endIdx, dataset.end());

        FrequencyTable table = train(trainingData);
        double percentage = test(table, testingData);

        cout << "Acuraccy on try " << tries++ << " " << 100 * percentage << " %" << endl;
        average += percentage;
    }
    cout << "Average " << 100 * average / 10 << " %" << endl;
}

int main()
{
    NaiveBayesClassifier naiveBayesClassifier = NaiveBayesClassifier("C:/Users/martinev/Desktop/ИИ проекти условия/HW 5 AI NAIVE BAYES CLASSIFIER/house-votes-84.data");
    naiveBayesClassifier.run();
    return 0;
}