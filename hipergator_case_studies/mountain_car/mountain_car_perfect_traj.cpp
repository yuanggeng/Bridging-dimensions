#include "POLAR/NeuralNetwork.h"
#include "flowstar-toolbox/Discrete.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	clock_t begin, end;
	begin = clock();

	string net_name = "nn_LDC1test";//argv[2]"";
	// string net_name2 = "LDC2_(0,0.07)";
	// string net_name3 = "LDC3_1st_(-0.6,0.6)(0,0.07)";
	string net_name1 = "LDC60_(-0.6, -0.55)-(0.0, 0.01)";
	string net_name2 = "LDC60_(-0.6, -0.55)-(0.01, 0.02)";
	string net_name3 = "LDC60_(-0.6, -0.55)-(0.02, 0.03)";
	string net_name4 = "LDC60_(-0.6, -0.55)-(0.03, 0.04)";
	string net_name5 = "LDC60_(-0.6, -0.55)-(0.04, 0.05)";
	string net_name6 = "LDC60_(-0.55, -0.5)-(0.0, 0.01)";
	string net_name7 = "LDC60_(-0.55, -0.5)-(0.01, 0.02)";
	string net_name8 = "LDC60_(-0.55, -0.5)-(0.02, 0.03)";
	string net_name9 = "LDC60_(-0.55, -0.5)-(0.03, 0.04)";
	string net_name10 = "LDC60_(-0.55, -0.5)-(0.04, 0.05)";




	string benchmark_name = "mountain_car";
	
	// Declaration of the state variables.
	unsigned int numVars = 3;

	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;
	
	/* Old inteface
	// Define the discrete dynamics.
    // x0 is the position of the mountain car, x1 is the speed of the mountain car.
	Expression<Interval> deriv_x0("x0 + x1", vars); // Discrete: Next_x0 = x0 + x1
	Expression<Interval> deriv_x1("x1 + 0.0015 * u - 0.0025 * cos(3 * x0)", vars); // Discrete: Next_x1 = x1 + 0.0015 * u - 0.0025 * cos(3 * x0)
	Expression<Interval> deriv_u("u", vars);

	vector<Expression<Interval> > dde_rhs(numVars);
	dde_rhs[x0_id] = deriv_x0;
	dde_rhs[x1_id] = deriv_x1;
	dde_rhs[u_id] = deriv_u;

	Nonlinear_Discrete_Dynamics dynamics(dde_rhs);
	*/
	// Define the discrete dynamics.
	DDE<Real> dynamics({"x0 + x1",
			    "x1 + 0.0015 * u - 0.0025 * cos(3 * x0)",
			    "0"}, vars);
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);
	//Computational_Setting setting;

	//unsigned int order = stoi(argv[4]);
	unsigned int order = 6;


	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.01, order); // the stepsize will be ignored

	// cutoff threshold
	setting.setCutoffThreshold(1e-10);

	// print out the steps
	setting.printOff();

/*	// DDE does not require a remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);
*/
	//setting.printOn();

	//setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	int steps = 60; //	int steps = 100;

	double start_x0 = std::atof(argv[1]);
    double end_x0 = std::atof(argv[2]);
    double start_x1 = std::atof(argv[3]);
    double end_x1 = std::atof(argv[4]);

    //-0.6 -0.4   -0.02, 0.07 // 0, 0.05
	for (double start_pos = start_x0; start_pos <= end_x0; start_pos += 0.01 ){
		for (double start_velocity = start_x1; start_velocity <= end_x1; start_velocity += 0.001){
	//double w = stod(argv[1]);
	//Interval init_x0(-0.515 - w, -0.515 + w), init_x1(0);
	Interval init_x0(start_pos, start_pos + 0.01), init_x1(start_velocity, start_velocity + 0.001);
	Interval init_u(0); // w=0.05
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
	vector<Constraint> safeSet;
	//vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	NeuralNetwork nn; //declare the nn outside of the if and else
	Interval CP(-0.01, 0.01);
	// if (-1.2 <= start_pos && start_pos + 0.01 <= 0.6 && start_velocity + 0.001 <= 0.07 && 0 <= start_velocity ){
	// 	string nn_name = net_name2;
	// 	nn = NeuralNetwork(nn_name);
	// 	//CP = Interval(-0.034, 0.034);
	// 	CP = Interval(-0.05, 0.05);

	// }else if (-0.6 <= start_pos && start_pos + 0.01 <= 0.6 && start_velocity + 0.001 <= 0.07 && 0 <= start_velocity){
	// 	string nn_name = net_name3;
	// 	nn = NeuralNetwork(nn_name);
	// 	//CP = Interval(-0.05, 0.05);
	// 	CP = Interval(-0.05, 0.05);
	// }
	// else {
	// 	string nn_name = net_name;
	// 	nn = NeuralNetwork(nn_name);
	// 	//CP = Interval(-0.021, 0.021);
	// 	CP = Interval(-0.05, 0.05);
	// }

	if (-0.6 <= start_pos && start_pos + 0.01 <= -0.55 && start_velocity + 0.001 <= 0.01 && 0 <= start_velocity ){
		string nn_name = net_name1;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else if (-0.6 <= start_pos && start_pos + 0.01 <= -0.55 && start_velocity + 0.001 <= 0.02 && 0.01 <= start_velocity){
		string nn_name = net_name2;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else if (-0.6 <= start_pos && start_pos + 0.01 <= -0.55 && start_velocity + 0.001 <= 0.03 && 0.02 <= start_velocity){ 
		string nn_name = net_name3;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else if (-0.6 <= start_pos && start_pos + 0.01 <= -0.55 && start_velocity + 0.001 <= 0.04 && 0.03 <= start_velocity){
		string nn_name = net_name4;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.04, 0.04);
	}
	else if (-0.6 <= start_pos && start_pos + 0.01 <= -0.55 && start_velocity + 0.001 <= 0.05 && 0.04 <= start_velocity){
		string nn_name = net_name5;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.03, 0.03);
	}
	else if (-0.55 <= start_pos && start_pos + 0.01 <= -0.5 && start_velocity + 0.001 <= 0.01 && 0 <= start_velocity){
		string nn_name = net_name6;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.02, 0.02);
	}
	else if (-0.55 <= start_pos && start_pos + 0.01 <= -0.5 && start_velocity + 0.001 <= 0.02 && 0.01 <= start_velocity){
		string nn_name = net_name7;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else if (-0.55 <= start_pos && start_pos + 0.01 <= -0.5 && start_velocity + 0.001 <= 0.03 && 0.02 <= start_velocity){
		string nn_name = net_name8;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else if (-0.55 <= start_pos && start_pos + 0.01 <= -0.5 && start_velocity + 0.001 <= 0.04 && 0.03 <= start_velocity){
		string nn_name = net_name9;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.03, 0.03);
	}
	else if (-0.55 <= start_pos && start_pos + 0.01 <= -0.5 && start_velocity + 0.001 <= 0.05 && 0.04 <= start_velocity){
		string nn_name = net_name10;
		nn = NeuralNetwork(nn_name);
		CP = Interval(-0.05, 0.05);
	}
	else {
		string nn_name = net_name;
		nn = NeuralNetwork(nn_name);
		//CP = Interval(-0.021, 0.021);
		CP = Interval(-0.05, 0.05);
	}
	//std::cout << "neural network that will be used:" << nn_name << endl;

	
	// string nn_name = "nn_"+net_name;
	// NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	//unsigned int bernstein_order = stoi(argv[3]);
	unsigned int bernstein_order = 4;
	unsigned int partition_num = 4000;

	//unsigned int if_symbo = stoi(argv[5]);
	unsigned int if_symbo = 1;

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x0");
	state_vars.push_back("x1");

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);

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


		Matrix<Interval> rm1(1, 1);
		tmv_output.Remainder(rm1);
		cout << "Previous Neural network taylor remainder: " << rm1 << endl;

		// Interval CP(-0.024, 0.024);
		cout << "CP value in this step: " << CP << endl;
		tmv_output.tms[0].remainder = tmv_output.tms[0].remainder + CP;

		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);
		//dynamics.reach_sr(result, setting, initial_set, 1, symbolic_remainder, unsafeSet);
		vector<Interval> end_box;
		result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
		std :: cout << "Upper and lower bound in the x0:" << end_box[0].sup()<< endl;
		std :: cout << "Upper and lower bound in the u:" << end_box[1]<< endl;
		// my definition for over-approximation
		if (end_box[0].sup() > 30){
			break;
		}

		// string nn_name;
		// if (-1.2 <= end_box[0].inf() && end_box[0].sup() <= 0.6 && end_box[1].sup() <= 0.07 && 0 <= end_box[1].inf() ){
		// 	nn_name = net_name2;
		// 	nn = NeuralNetwork(nn_name);
		// 	//CP = Interval(-0.034, 0.034);
		// 	CP = Interval(-0.1, 0.1);


		// }else if (-0.6 <= end_box[0].inf() && end_box[0].sup() <= 0.6 && end_box[1].sup() <= 0.07 && 0 <= end_box[1].inf()){
		// 	nn_name = net_name3;
		// 	nn = NeuralNetwork(nn_name);
		// 	//CP = Interval(-0.05, 0.05);
		// 	CP = Interval(-0.1, 0.1);


		// }else {
		// 	nn_name = net_name;
		// 	nn = NeuralNetwork(nn_name);
		// 	//CP = Interval(-0.0209, 0.0209);
		// 	CP = Interval(-0.1, 0.1);

		// }
		// std::cout << "neural network that will be used:" << nn_name << endl;



		// not using a symbolic remainder
		// dynamics.reach(result, setting, initial_set, 1, unsafeSet);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			std::string bad_estimation = "overest";
			ofstream result_output("./outputs_perf_traj/test_overestimation/" + bad_estimation  + "_" + to_string(start_pos) +to_string(start_velocity) +  ".txt");
			//return 1;
		}
	}


	vector<Constraint> targetSet;
	Constraint c1("-x0 + 0.45", vars);		// x0 >= 0.2
	Constraint c2("-x1 + 0.0", vars);		// x0 >= 0.0

	targetSet.push_back(c1);
	targetSet.push_back(c2);

	string reach_result;

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);

	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
		std::string good_result = "Yes";
		ofstream reach_output("./outputs_perf_traj/reach_verification/" + good_result + "_" + to_string(start_pos) + to_string(start_velocity) +".txt");
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(steps) + ")";
	}


	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("x1", "x0");

	int mkres = mkdir("./outputs_perf_traj", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs_perf_traj/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs_perf_traj/", benchmark_name + "_" + to_string(if_symbo) + "_" + to_string(start_pos) + "_" + to_string(start_velocity), result.tmv_flowpipes, setting);
	//plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result);
		}
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	double exe_time;
	exe_time = (double)(end - begin) / CLOCKS_PER_SEC;
	ofstream reach_output("./outputs_perf_traj/computation_time/" + to_string(exe_time) +".txt");
	return 0;
}
