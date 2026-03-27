#include <iostream>
#include <vector>

using namespace std;

class LinearModel{
private:
    vector<vector<double>> X;
    vector<double> target;
    vector<double> weights;
    double bias;
    double learning_rate;
    int num_iterations;

public:
    LinearModel(){
        X = vector<vector<double>>();
        target = vector<double>();
        weights = vector<double>();
        bias = 0.0;
        learning_rate = 0.01;
        num_iterations = 1000;
    }

    LinearModel(vector<vector<double>> input_X, vector<double> Y){
        X = input_X;
        target = Y;
        weights = vector<double>();
        bias = 0.0;
        learning_rate = 0.01;
        num_iterations = 1000;
    }

    double predict(const vector<double>& x){
        if (x.size() != weights.size()){
            cout << "Error: Размерность входного вектора не соответствует размеру вектора весов." << endl;
            return 0.0;
        }

        double result = bias;

        for (int j = 0; j < x.size(); j++){
            result += weights[j] * x[j];
        }

        return result;
    }   

    void fit(){
        if (X.empty() || target.empty()){
            cout << "Error: Данные для обучения не предоставлены." << endl;
            return;
        }

        if (X.size() != target.size()){
            cout << "Error: Количество образцов в X не соответствует количеству меток в target." << endl;
            return;
        }

        int num_features = X[0].size();

        for (int i = 0; i < X.size(); i++){
            if (X[i].size() != num_features){
                cout << "Error: Все образцы в X должны иметь одинаковое количество признаков." << endl;
                return;
            }
        }

        weights = vector<double>(num_features, 0.0);
        bias = 0.0;

        for (int iter = 0; iter < num_iterations; iter++){
            vector<double> dw(num_features, 0.0);
            double db = 0.0;
            
            for (int i = 0; i < X.size(); i++){
                double y_pred = predict(X[i]);
                double error = y_pred - target[i];

                for (int j = 0; j < num_features; j++){
                    dw[j] += error * X[i][j];
                }

                db += error;
            }

            for (int j = 0; j < num_features; j++){
                dw[j] /= X.size();
                weights[j] -= learning_rate * dw[j];
            }

            db /= X.size();
            bias -= learning_rate * db;

            if (iter % 100 == 0){
                cout << "Iteration " << iter << ": Weights = [";
                for (int j = 0; j < num_features; j++){
                    cout << weights[j];
                    if (j < num_features - 1){
                        cout << ", ";
                    }
                }
                cout << "], Bias = " << bias << endl;
            }
        }
    }
};

int main(){
    vector<vector<double>> X = {
        {1.0, 2.0},
        {2.0, 1.0},
        {3.0, 0.0},
        {0.0, 3.0}
    };
    vector<double> Y = {5.0, 4.0, 3.0, 6.0};

    LinearModel model(X, Y);
    model.fit();

    return 0;
}