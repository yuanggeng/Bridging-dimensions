#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

#include <typeinfo>
#include <iostream>


using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	clock_t begin, end;
	begin = clock();
	//string net_name = "controller_single_pendulum_POLAR";
    string net_name = "POLAR_retrain_20_LDC1_theta0-6.28_dot0-8_wholestate.txt";


	string benchmark_name = "Single_Pendulum";
	// Declaration of the state variables.
	unsigned int numVars = 4;

//	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int t_id = vars.declareVar("t");
	int u_id = vars.declareVar("u0");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x1","2*sin(x0) + 8*u0","1","0"}, vars);
	//ODE<Real> dynamics({"x1","9.81*sin(x0)+u0","1","0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.02, order);


	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	//cout << "initial remainder: " << I << endl;

	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

  //	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	// 0, 2; -2, 0//0,1 + -2,-1
	for (double start = 0; start <= 1; start += 0.01){
	int steps = 30;
	//float startpoint = 1.3;
	float startpoint = floorf(start * 100) / 100;
	double  endpoint = startpoint + 0.01;
	for (double startx1 = -2; startx1<= -1; startx1+=0.01){
	//Interval init_x0(startpoint, endpoint), init_x1(0.0,0.0), init_t(0), init_u(0);
	Interval init_x0(startpoint, endpoint), init_x1(startx1, startx1 + 0.01), init_t(0), init_u(0);

	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_t);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;

	unsigned int bernstein_order = 2;
	unsigned int partition_num = 100;

	unsigned int if_symbo = 1;

	double err_max = 0;


	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	clock_t begin, end;
	begin = clock();

	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.  " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);

		// TaylorModelVec<Real> tmv_temp;
		// initial_set.compose(tmv_temp, order, cutoff_threshold);
		// tmv_input.tms.push_back(tmv_temp.tms[0]);
		// tmv_input.tms.push_back(tmv_temp.tms[1]);


		// taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		TaylorModelVec<Real> tmv_output;

		if(if_symbo == 0){
			// not using symbolic remainder
			nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else{
			// using symbolic remainder
			nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		//Yuang's pathetic invasion

		//Interval rm2(1,1); 
		/*
		Matrix<Interval> rm1(1, 1);
		Matrix<double> rm2(1, 1, 0.1);
		cout << "rm1 matrix: " << rm1 << endl;
		cout << "rm2 matrix: " << rm2 << endl;
		cout << "Previous Neural network taylor remainder: " << rm1 + rm2 << endl;
		tmv_output.Remainder(rm1);
		//std::cout << typeid(rm1).name() << std::endl;
		cout << "After Neural network taylor remainder: " << rm1 << endl; */

		//Interval test_1(1,1);
		//cout << "test result: " << test_1  << endl;
		//rm1 = rm1.add_assign(0.01)
		//std::valarray<double> new_array = tmv_output.tms[0] + 1;
		//initial_set.tmvPre.tms[u_id] = tmv_output.tms[0] + 0.1;


/*
		Interval CP_value(0.1, 0.1);
		tmv_output.tms[0].remainder = tmv_output.tms[0].remainder + CP_value;
		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];
*/
 	    Interval box;
        tmv_output.tms[0].intEval(box, initial_set.domain);
        cout << "Previous neural network output range by TMP: " << box << endl;
		std::cout << "The remainder in this step: " << tmv_output.tms[0].remainder << std::endl;
		//Interval CP(-0.042, 0.042);
		//std::cout << "After CP, the remainder in the calculation: " << tmv_output.tms[0].remainder + CP << std::endl;


		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		//Interval CP(-0.025, 0.025);
		//tmv_output.tms[0].remainder = tmv_output.tms[0].remainder + CP;
		std::cout << "After CP, the remainder is: " << tmv_output.tms[0].remainder << std::endl;

		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

		dynamics.reach(result, initial_set, 0.05, setting, safeSet, symbolic_remainder);
		vector<Interval> end_box;
		result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
		std :: cout << "upper and lower bound in the x0" << end_box[0]<< endl;

		//Print the upper and lower bound
		//tmvPre_test = result.fp_end_of_time.tmvPre;
		//std::vector<Interval> tmvRange_test;
		//tmvPre_test.intEvalNormal(tmvRange_test, tm_setting.step_exp_table);
		//result.fp_end_of_time.tmvPre.intEvalNormal(tmvRange_test, tm_setting.step_exp_table);


		//Invasion: test safety every steps here
		/*
		bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
		vector<Interval> end_box;
		result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
		std :: cout << "upper and lower bound in the x0" << end_box[0]<< endl;
		std :: cout << "upper and lower bound in the x1:" << end_box[1]<< endl;
		std :: cout << "upper and lower bound in the output u:" << end_box[3]<< endl;


        string reach_result;

        if(b)
        {
                reach_result = "Verification result: Yes(" + to_string(steps) + ")";
		        std::string good_result = "Yes";
		        ofstream reach_output("./outputs/reach_verification/" + good_result + "_" + to_string(startpoint) + to_string(startx1) +".txt");
		        cout << reach_result << endl;
        }
        else
        {
                reach_result = "Verification result: No(" + to_string(steps) + ")";
        }
	
*/


		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;

			//cout << "Flowpipe taylor expansion: " << initial_set.tmv.tms[0].expansion.coefficient << endl;//"     " << initial_set.tmv.tms[1].expansion << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			std::string bad_estimation = "overest";
			ofstream result_output("./outputs/test_overestimation/" + bad_estimation  + "_" + to_string(startpoint) +to_string(startx1) +  ".txt");
			//return 1;
		}
	}

	vector<Constraint> unsafeSet = {Constraint("-t + 0.5", vars), Constraint("-x0 + 6", vars), Constraint("x0 - 6.2", vars)};
        //vector<Constraint> unsafeSet = {Constraint("-t + 0.5", vars), Constraint("-x0 + 10", vars)};
	result.unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);

	if(result.isUnsafe())
	{
		printf("The system is unsafe.\n");
	}
	else if(result.isSafe())
	{
		printf("The system is safe.\n");
	}
	else
	{
		printf("The safety is unknown.\n");
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

	//target set
	vector<Constraint> targetSet;
        //Constraint c1("x0 - 6.5", vars);          // x0 <= 6.2
        //Constraint c2("-x0 + 5.9", vars);   
		Constraint c3("x0 - 0.35", vars);      
	    Constraint c4("-x0 + 0", vars);           //x0 >= 0 can only capture the first two constrants
        //Constraint c5("x1 - 1", vars);            // x1 <= 1
        //Constraint c6("-x1 + 1", vars);           // x1 >= 1

        //targetSet.push_back(c1);
        //targetSet.push_back(c2);
	    targetSet.push_back(c3);
	    targetSet.push_back(c4);
        //targetSet.push_back(c5);
        //targetSet.push_back(c6);

        
        bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
        string reach_result;

        if(b)
        {
        reach_result = "Verification result of target set: Yes(" + to_string(steps) + ")";
		std::string good_result = "Yes";
		ofstream reach_output("./outputs/reach_verification/" + good_result + "_" + to_string(startpoint) + to_string(startx1) +".txt");
        }
        else
        {
                reach_result = "Verification result of target set: No(" + to_string(steps) + ")";
        }
	cout << reach_result << endl;

	//Yuang another pathetic invasion
	cout << "try to print the last reachable set size: " << result.tmv_flowpipes.size()<< endl;



	// our printing
	/* 
	flowstar::Flowpipe fp = result.fp_end_of_time;
	for (flowstar::TaylorModel<Real> tm: fp.tmv.tms) {
		for (Term<Real> term: tm.expansion.terms ){ 
			printf("In term loop\n");
			string s("sadaskdjafjew fwjefwkjefwkejnfwkjenfkewj nfkewjnfkwnfkwej nfkewjfnkjew nfkewnfk "); 
			term.toString(s, vars);
			printf("Term: %s\n", s.c_str());
			// printf("Term coefficient: %lf\n", term.coefficient);

		}
		printf("Remainder: [%lf, %lf]\n", tm.remainder.inf(), tm.remainder.sup()); 
		// tm.remainder
		// printf("Result: %d", 4); 
	}
	printf("Result: %d", 4); 
	*/

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("t", "x0");

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	//print the safet results if you want.
	//ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + "_" + to_string(startpoint) +  ".txt");

	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo) + "_" + to_string(startpoint) + "_" + to_string(startx1) , result.tmv_flowpipes, setting);
	}
	}

	//Stupid Yuang want to extract the upper and lower bound of the reachable set
	/*std::ifstream file("outputs/Single_Pendulum_1_0.290000_0.400000.plt");

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {  // Read the file line by line
        std::cout << line << std::endl;  // Print each line
    }

    // Close the file
    file.close(); */
    
	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	double exe_time;
	exe_time = (double)(end - begin) / CLOCKS_PER_SEC;
	ofstream reach_output("./outputs/computation_time/" + to_string(exe_time) +".txt");

	return 0;
	
}
