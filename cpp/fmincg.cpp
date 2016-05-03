#include "fmincg.h"
#include <cstdio>

#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define color_yellow(str)		{ ANSI_COLOR_YELLOW		str ANSI_COLOR_RESET }

using namespace arma;

void fmincg2(
    double& finalcost, const int length,
    CostFunction costfunction, mat& nn_params) { //,const int input_layer_size, const int hidden_layer_size, const int num_labels, mat& inputdata, mat& y, const double lambda){
// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
// (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
// 
// Permission is granted for anyone to copy, use, or modify these
// programs and accompanying documents for purposes of research or
// education, provided this copyright notice is retained, and note is
// made of any changes that have been made.
// 
// These programs and documents are distributed without any warranty,
// express or implied.  As the programs were written for research
// purposes only, they have not been tested to the degree that would be
// advisable in any important application.  All use of these programs is
// entirely at the user's own risk.
//
// [ml-class] Changes Made:
// 1) Function name and argument specifications
// 2) Output display
//
// [John Shahbazian] Changes Made:
// 1) Ported to C++ using the Armadillo (http://arma.sourceforge.net/) library
// 2) Change the cost function call to internal.  Replace the
//    'costfunction' function to whatever you would like.  It returns
//    the cost as the result, and the gradient is returned through the first 
//    argument as an intent(inout) (e.g. 'gradient#').  
// 3) Changed the variable names to be readable.
//    f1 = cost1
//    df1 = gradient1
//    s = search_direction
//    d1 = slope1
//    z1 = point1
//    X0 = backup_params
//    f0 = cost_backup
//    df0 = gradient_backup 
//
// [Tim Yong] Changes made:
// 1) allow the usage of CostFunction input
// 2) remove the extraneous parameters from the function call input

	const double RHO = 0.01;
	const double SIG = 0.5;
	const double INT = 0.1;
	const double EXT = 3.0;
	const int MAXEVALS = 20;
	const double RATIO = 100;
	
	double mintemp, minstuff, M, A, B;
	double fX = 0.0;
	int success;
	int i=0;
	int ls_failed = 0;
	
	mat backup_params = nn_params;
	mat gradient2(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat gradient3(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat gradient_backup(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat tmp(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	double limit;
	double point1, point2, point3;
	double cost1, cost2, cost3, cost_backup;
	double slope1, slope2, slope3;
	mat search_direction(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	mat stemp(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	
	double sqrtnumber;
	//mat sd_calc_1,sd_calc_2,sd_calc_3;
	//double sd_calc_4;
	//mat sd_calc_5(nn_params.n_rows,nn_params.n_cols,fill::zeros);

	
	cost1 = 10000.0;  //lower is better, so init with high
	mat gradient1(nn_params.n_rows,nn_params.n_cols,fill::zeros);
	costfunction(cost1, gradient1, nn_params);//, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
	//std::cout << "gradient1: " << gradient1(0,0) << endl;
	//pause();

	//i = i + (length<0);
	search_direction = -gradient1;

	mat slope_vector(1,1);
	slope_vector = -search_direction.t() * search_direction;
	slope1 = slope_vector(0,0);
	point1 = 1.0/(1.0 - slope1);
	//std::cout << "point1: " << point1 << endl;

	while(i < std::abs(length)){
		i = i + 1;
		//std::cout << "loop: " << i << endl;
		backup_params = nn_params;
		cost_backup = cost1;
		gradient_backup = gradient1;
		stemp = point1 * search_direction;
		nn_params = nn_params + stemp;
		//std::cout << "nn_params: " << nn_params.row(0) << endl;

		gradient2 = gradient1;
		
		cost2 = 10000.0;
		static int iter;
		printf(color_yellow("Iteration #%d.1\n"), iter);
		costfunction(cost2, gradient2, nn_params);//, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
		
		//i = i + (length<0);
		//i++;
		//std::cout << "i: " << i << endl;
		slope_vector = gradient2.t() * search_direction;
		slope2 = slope_vector(0,0);
		//std::cout << "slope2: " << slope2 << endl;


		cost3 = cost1;
		slope3 = slope1;
		point3 = -point1;
		if(length>0)M=MAXEVALS;
		else M = std::min(MAXEVALS,-length-i);
		success=0;
		limit=-1;
		
		while(1){ //3.66252   3.66252  0 -5.40439
			//std::cout << "cost2: " << cost2 << " cost1: " << cost1 << " point1: " << point1 << " slope1: " << slope1 << endl;
			//std::cout << "cost2 vs stuff: " << cost2 << "   " << (cost1 + (point1 * RHO * slope1)) << endl;
			//std::cout << "slope2 vs stuff: " << slope2 << "   " << (-SIG * slope1) << endl;
			//std::cout << "M: " << M << endl;
			//pause();
			while(( (cost2 > (cost1 + (point1 * RHO * slope1))) || (slope2 > (-SIG * slope1)) ) && (M > 0) ){
				//std::cout << "here*******************" << endl;
				limit = point1;
				if(cost2 > cost1)
                    point2 = point3 - (0.5 * slope3 * point3 * point3)/(slope3 * point3 + cost2 - cost3);  //quadratic fit
                else{
                    A = 6*(cost2 - cost3)/point3 + 3*(slope2 + slope3);           //cubic fit
                    B = 3*(cost3 - cost2) - point3 * (slope3 + 2*slope2);
                    point2 = (std::sqrt(B*B - A*slope2*point3*point3) - B)/A;
				}
                if(std::isnan(point2) || (!std::isfinite(point2)))
                    point2 = point3 / 2;                         // if we had a numerical problem then bisect
                point2 = std::max( std::min(point2, (INT * point3)), ((1.0 - INT) * point3));  //don't accept too close to limits
                point1  = point1 + point2;                       // update the step
                stemp = point2 * search_direction;
                nn_params = nn_params + stemp;

								printf(color_yellow("Iteration #%d.2\n"), iter);
                costfunction(cost2, gradient2, nn_params);//, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);

                M = M - 1.0;
				//std::cout << "i2: " << i << endl;
                //i = i + (length<0);                              // count epochs?!
				//std::cout << "i2: " << i << endl;
                slope_vector = gradient2.t() * search_direction;
                slope2 = slope_vector(0,0);                      //convert to scalar
                point3 = point3 - point2;                        // point3 is now relative to the location of point2                

			}
			
            if ((cost2 > (cost1 + (point1*RHO*slope1)) ) || (slope2 > (-SIG * slope1) )){
				//std::cout << "Break -----------------" << endl;
				break;                                            // this is a failure

            }
            else if (slope2 > (SIG * slope1)){
				//std::cout << "Break -----------------" << endl;
                success = 1;
                break;                                            // success
			}
            else if (M == 0.0){
				//std::cout << "Break -----------------" << endl;
                break;                                            // failure
			}

            A = 6*(cost2 - cost3)/point3 + 3*(slope2 + slope3);  // make cubic extrapolation
            B = 3*(cost3 - cost2) - point3*(slope3 + 2*slope2);
            sqrtnumber = (B*B) - A*slope2*point3*point3;
            
			if((!std::isnormal(sqrtnumber)) || (sqrtnumber < 0.0) ){
				//std::cout << "sqrt ifs" << endl;
                if (limit < -0.5)                          // if we have no upper limit
                    point2 = point1  * (EXT - 1);                // the extrapolate the maximum amount
                else
                    point2 = (limit - point1) / 2;               // otherwise bisect
				//std::cout << "point2 sqrt ifs: " << point2 << endl;
            }
            else{
                point2 = (-slope2 * point3 * point3)/(B + std::sqrt(sqrtnumber));
                if ((limit > -0.5) && ((point2 + point1) > limit))          // extraplation beyond max?
                    point2 = (limit - point1)/2;                 // bisect
                else if ((limit < -0.5) && ((point2 + point1) > (point1 * EXT)))       // extrapolation beyond limit
                    point2 = point1 * (EXT - 1.0);               // set to extrapolation limit
                else if (point2 < (-point3 * INT))
                    point2 = -point3 * INT;
                else if ((limit > -0.5) && (point2 < (limit - point1)*(1.0 - INT)))   // too close to limit?
                    point2 = (limit - point1 ) * (1.0 - INT);
            }			
			
            cost3 = cost2;
            slope3 = slope2;
            point3 = -point2;               
            point1  = point1 + point2;

			//std::cout << "search_direction: " << endl << search_direction.rows(0, 10) << endl;
            stemp = point2 * search_direction;
			//std::cout << "point2: " << point2 << endl;

            nn_params = nn_params + stemp;                       // update current estimates
			//std::cout << "nn_paramsb: " << endl << nn_params.rows(0,10) << endl;

						printf(color_yellow("Iteration #%d.3\n"), iter);
            costfunction(cost2, gradient2, nn_params);//, input_layer_size, hidden_layer_size, num_labels, inputdata, y, lambda);
			//std::cout << "cost2b: " << cost2 << endl;
            M = M - 1.0;
			//std::cout << "i: " << i << endl;
            //i = i + (length<0);                                  // count epochs?!
			//std::cout << "i: " << i << endl;
			slope_vector = gradient2.t() * search_direction;
            slope2 = slope_vector(0,0);                          //convert to scalar
			//std::cout << "slope2b: " << slope2 << endl;
			//pause();
		}
		
        if (success == 1){                                   // if line search succeeded
            cost1 = cost2;
            fX = cost1;
            //std::cout << "Iteration: " << i << " | Cost: " << cost1 << endl;
            mat sd_calc_1 = gradient2.t() * gradient2;
            mat sd_calc_2 = gradient1.t() * gradient2;
            mat sd_calc_3 = gradient1.t() * gradient1;
            double sd_calc_4 = (sd_calc_1(0,0) - sd_calc_2(0,0)) / sd_calc_3(0,0);
            mat sd_calc_5 = sd_calc_4 * search_direction;
            
            search_direction = sd_calc_5 - gradient2;
            tmp = gradient1;
            gradient1 = gradient2;
            gradient2 = tmp;                                     // swap derivatives
			slope_vector = gradient1.t() * search_direction;
            slope2 = slope_vector(0,0);                          //convert to scalar
            if(slope2 > 0.0){                                  // new slope must be negative
                search_direction = -gradient1;                   // otherwise use steepest direction
				slope_vector = -search_direction.t() * search_direction;
                slope2 = slope_vector(0,0);                      //convert to scalar
            }
			mintemp = slope1 / (slope2);// - std::numeric_limits<double>::lowest());  //std::numeric_limits<double>::lowest() is min value double precision float //TODO: figure out why the min number is needed
            minstuff = std::min(RATIO, mintemp);
            point1  = point1 * minstuff;                         // slope ratio but max RATIO
            slope1 = slope2;
            ls_failed = 0;                                        // this line search did not fail
        }
        else{
            nn_params = backup_params;
            cost1 = cost_backup;
            gradient1 = gradient_backup;                         // restore point from before failed line search
            if (ls_failed == 1 || (i > std::abs(length)))       // line search failed twice in a row
				//std::cout << "Break -----------------" << endl;
                break;                                            // or we ran out of time, so we give up
            tmp = gradient1;
            gradient1 = gradient2;
            gradient2 = tmp;                                    // swap derivatives
            search_direction = -gradient1;                      // try steepest
            slope_vector = -search_direction.t() * search_direction;
            slope1 = slope_vector(0,0);                         // convert to scalar
            point1  = 1.0 / (1.0 - slope1);
            ls_failed = 1;                                      // this line search failed
        }
				iter++;
       
	}
	
	finalcost = fX;		//return finalcost


	//std::cout << "new nn params: \n";
	//std::cout << nn_params << endl;
}
