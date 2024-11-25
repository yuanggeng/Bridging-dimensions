#include "POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	clock_t begin, end;
	begin = clock();

	//string net_name = "Cart_LDC1_whole_5layers";
	// string net_name = "test_nn_action0";
	// string net_name = "test_nn_action0";

	string net_name = "Cart_LDC1_3layers_new";

	string benchmark_name = "Double_Pendulum_more";
	// Declaration of the state variables.
	//unsigned int numVars = 7; //Default
	unsigned int numVars = 6;

//	intervalNumPrecision = 600;

	Variables vars;
	// Probably wrong old version
	// int x0_id = vars.declareVar("x0"); //pos x
	// int x1_id = vars.declareVar("x1"); //theta
    // int x2_id = vars.declareVar("x2"); //dpos
    // int x3_id = vars.declareVar("x3"); // dtheta
    // int t_id = vars.declareVar("t");
	// int u0_id = vars.declareVar("u0"); //force on the cart

	// new test failed
	int x0_id = vars.declareVar("x0"); // pos x
	int x1_id = vars.declareVar("x1"); // velcocity
    int x2_id = vars.declareVar("x2"); // theta
    int x3_id = vars.declareVar("x3"); // dtheta
    int t_id = vars.declareVar("t");
	int u0_id = vars.declareVar("u0"); //force on the cart
	int domainDim = numVars + 1;

    // Old version
	// string thetaacc = "(9.8 * sin(x1) - cos(x1) * ((u0 + 0.05 * x3 * x3 * sin(x1)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x1) * cos(x1) / 1.1))";
	// string xacc = "(((u0 + 0.05 * x3 * x3 * sin(x1)) / 1.1) - (0.05 * (9.8 * sin(x1) - cos(x1) * ((u0 + 0.05 * x3 * x3 * sin(x1)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x1) * cos(x1) / 1.1)) * cos(x1)) / 1.1)";
    // ODE<Real> dynamics({"x2", "x3", xacc, thetaacc, "1","0"}, vars);

	// new test: failed at high theta
	string thetaacc = "(9.8 * sin(x2) - cos(x2) * ((u0 + 0.05 * x3 * x3 * sin(x2)) / 1.1)) /(0.5 * (4.0/3.0 - 0.1 * cos(x2) * cos(x2) / 1.1))";
	string xacc = "((u0 + 0.05 * x3 * x3 * sin(x2)) / 1.1) - 0.05 * ((9.8 * sin(x2) - cos(x2) * ((u0 + 0.05 * x3 * x3 * sin(x2)) / 1.1)) /(0.5 * (4.0/3.0 - 0.1 * cos(x2) * cos(x2) / 1.1))) * cos(x2) / 1.1";
    ODE<Real> dynamics({"x1", xacc, "x3", thetaacc, "1", "0"}, vars);


	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.01, order);

	// time horizon for a single control step
//	setting.setTime(0.5);

	// cutoff threshold
	setting.setCutoffThreshold(1e-6);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

//	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = 0.01;
	int steps = 20;
	// for loop for all the 4 states
	// for (double pos_start = 0; pos_start <= 0.1; pos_start += 0.01){

	// 	for (double vel_start = 0; vel_start <= 0.05; vel_start += 0.01){

	// 				for (double theta = 0.15; theta <= 0.25; theta += 0.01){

	// 							for (double dot = 0.05; dot <= 0.1; dot += 0.01){

	// 0, 0.1 x 0, 0.1 x 0.06, 0.1 x -0.4 -0.35

	double start_pos = std::atof(argv[1]);
    double end_pos = std::atof(argv[2]);
    double start_vel = std::atof(argv[3]);
    double end_vel = std::atof(argv[4]);
	double start_the = std::atof(argv[5]);
    double end_the = std::atof(argv[6]);
	double start_dot = std::atof(argv[7]);
    double end_dot = std::atof(argv[8]);


	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	for (double pos = start_pos; pos < end_pos; pos += 0.01){

		for (double vel = start_vel; vel < end_vel; vel += 0.01){

					for (double theta = start_the; theta <= end_the; theta += 0.01){
					// for (double theta = 0.06; theta <= 0.16; theta += 0.01){

								for (double dot = start_dot; dot < end_dot; dot += 0.01){
	

	//new test: failed
	// Interval init_x0(pos_start - w, pos_start + w), init_x1(vel_start-w, vel_start + w), init_x2(theta-w, theta + w), init_x3(dot - w, dot + w), init_t(0), init_u0(0); //w=0.01
	Interval init_x0(pos, pos + w), init_x1(vel, vel + w), init_x2(theta, theta + w), init_x3(dot, dot + w), init_t(0), init_u0(0); //w=0.01

	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_t);
	X0.push_back(init_u0);

	//OLD version
	// Interval init_x0(pos_start - w, pos_start + w), init_x2(vel_start - w, vel_start + w), init_x1(theta - w, theta + w), init_x3(dot -w, dot + w), init_t(0), init_u0(0); //w=0.01
	// std::vector<Interval> X0;
	// X0.push_back(init_x0);
	// X0.push_back(init_x1);
	// X0.push_back(init_x2);
    // X0.push_back(init_x3);
    // X0.push_back(init_t);
	// X0.push_back(init_u0);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
    // safe set: when t>=0.4, all the veriables [-0.5, 1.5]
	// vector<Constraint> safeSet = {Constraint("x0 - 1.5", vars), Constraint("x1 - 1.5", vars), Constraint("x2 - 1.5", vars),
	// 		Constraint("x3 - 1.5", vars), Constraint("-x0 - 0.5", vars), Constraint("-x1 - 0.5", vars), Constraint("-x2 - 0.5", vars),
	// 		Constraint("-x3 - 0.5", vars)};

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


	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}
	// perform 35 control steps
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);
        tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);
        tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);

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


		Interval CP(-0.07, 0.07); //Interval CP for the who
		tmv_output.tms[0].remainder = tmv_output.tms[0].remainder + CP;


//		Matrix<Interval> rm1(1, 1);
//		tmv_output.Remainder(rm1);
//		cout << "Neural network taylor remainder: " << rm1 << endl;


		initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
        //initial_set.tmvPre.tms[u1_id] = tmv_output.tms[1];

		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, initial_set, 0.02, setting, safeSet, symbolic_remainder); //the time step is still 0.02

		// print the reachable set at each time step
		vector<Interval> end_box;
		result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
		// std :: cout << "upper and lower bound in the x0" << end_box[0]<< endl;
		// std :: cout << "upper and lower bound in the x1:" << end_box[1]<< endl;
		// std :: cout << "upper and lower bound in the x2:" << end_box[2]<< endl;
		// std :: cout << "upper and lower bound in the x3:" << end_box[3]<< endl;
		// std :: cout << "upper and lower bound in the t:" << end_box[4]<< endl;
		std :: cout << "upper and lower bound in the output u:" << end_box[5]<< endl;


		// if(result.status == COMPLETED_UNSAFE)
		// {
		// 	printf("The system is unsafe.\n");
		// 	break;
		// }

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
//			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			std::string bad_estimation = "overest";
			// ofstream result_output("./outputs/test_overestimation/" + bad_estimation  + "_" + to_string(startpoint) +to_string(startx1) +  ".txt");
			ofstream result_output("./outputs_perfect_act/test_overestimation/" + bad_estimation  + "_" + to_string(pos) +"_"+ to_string(vel)+"_"+ to_string(theta)+"_"+to_string(dot)+".txt");

		}
	}
		vector<Constraint> targetSet;
		Constraint c3("x2 - 0.2", vars);  //x2 < =0.2    
	    Constraint c4("-x2 - 0.2", vars);  
	    targetSet.push_back(c3);
	    targetSet.push_back(c4);


        
        bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
        string reach_result;

        if(b)
        {
        reach_result = "Verification result: Yes(" + to_string(steps) + ")";
		std::string good_result = "Yes";
		//ofstream reach_output("./outputs/reach_verification/" + good_result + "_" + to_string(startpoint) + to_string(startx1) +".txt");pos_start vel_start theta dot
		ofstream reach_output("./outputs_perfect_act/reach_verification/" + good_result + "_" + to_string(pos) +"_"+ to_string(vel) +"_"+ to_string(theta) +"_"+ to_string(dot)+".txt");

        }
        else
        {
            reach_result = "Verification result: No(" + to_string(steps) + ")";
        }


	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	//plot_setting.setOutputDims("x2", "x3");
	//plot_setting.setOutputDims("t", "x1");
	// plot_setting.setOutputDims("t", "x0");
	// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(pos_start) +"_"+ "x0" + "_" + to_string(vel_start)+"_"+ to_string(theta)+"_"+to_string(dot), result.tmv_flowpipes, setting);
	
	// plot_setting.setOutputDims("t", "x0");
	// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_"  + "pos" + "_" + to_string(pos) +"_"+ to_string(vel) +"_"+ to_string(theta) +"_"+ to_string(dot), result.tmv_flowpipes, setting);


	// plot_setting.setOutputDims("t", "x1");
	// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_"  + "velocity" + "_" + to_string(pos) +"_"+ to_string(vel) +"_"+ to_string(theta) +"_"+ to_string(dot), result.tmv_flowpipes, setting);


	// plot_setting.setOutputDims("t", "x3");
	// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_"  + "dot" +  "_" + to_string(pos) +"_"+ to_string(vel) +"_"+ to_string(theta) +"_"+ to_string(dot), result.tmv_flowpipes, setting);

	
	plot_setting.setOutputDims("t", "x2");
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs_perfect_act/", benchmark_name + "_" + "theta" + "_" + to_string(pos) +"_"+ to_string(vel) +"_"+ to_string(theta) +"_"+ to_string(dot), result.tmv_flowpipes, setting);
	}
					}
		}
	}
	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
