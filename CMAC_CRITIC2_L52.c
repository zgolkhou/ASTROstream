/*  File    : CMAC_CRITIC2_L52.c
/*  This S_function is a CMAC(Cerebullar Modeling Articulation Controller) Network. 
/*  It play a role of Critic at system.                                        */ 

#define S_FUNCTION_NAME CMAC_CRITIC2_L52
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include "stdio.h"

#define  U(element) (*uPtrs[element])  

static real_T mins[4] = {-0.2,-0.2,0,0};
static real_T sizes[4] = {0.4,0.4,15,15};
static real_T incr[4] = {0.04,0.04,1.5,1.5};
static real_T off[4] = {0.002,0.002,1.5/20,1.5/20};
		   	
/*====================
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        return; /* Parameter mismatch will be reported by Simulink */
    }

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 200081);     

    if (!ssSetNumInputPorts(S, 1)) return;  /* I have one input port*/
    ssSetInputPortWidth(S, 0, 5);           /* the width of input port is 5 */
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    if (!ssSetNumOutputPorts(S, 1)) return; /* I have one output port*/
    ssSetOutputPortWidth(S, 0, 2);          /*  the width of output port is 2  */

    ssSetNumSampleTimes(S, 1);
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    /* Take care when specifying exception free code - see sfuntmpl_doc.c */
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}

/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    Specifiy that we inherit our sample time from the driving block.
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, INHERITED_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);
}

#define MDL_INITIALIZE_CONDITIONS
/* Function: mdlInitializeConditions ========================================
 * Abstract:
 *    Initialize both discrete states to one.
 */
static void mdlInitializeConditions(SimStruct *S)
{
    real_T *x0 = ssGetRealDiscStates(S);
    int_T  lp,nDisc;
	nDisc = ssGetNumDiscStates(S);
    /* lp is number of dicrete state that all their initialize to 0 */  
    for (lp=0;lp<nDisc;lp++) { 
        *(x0+lp) = 0.0; 
    }
}



/* Function: mdlOutputs =======================================================
 * Abstract:
 *      y = Cx + Du 
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    real_T            *y    = ssGetOutputPortRealSignal(S,0);
    real_T            *x    = ssGetRealDiscStates(S);
    InputRealPtrsType uPtrs = ssGetInputPortRealSignalPtrs(S,0); /* second argument is 0 */
int_T  tiling=20,n=10,DIM=4,RES=20;
int_T  Size_state_Q=200000, Size_state_extra_Q=200080;
real_T alpha=0.7,gamma=0.95;
int_T  jp2=0,jp3=0,ij,ii,a_optim[20][4],Locate_Q_optim,y_optim,DIMc,k0,k1,k2,k3;
real_T minn,maxx,inc,mina2,ifloat2,ifloat3,Q_table_optim[20][20];
real_T mina3,maxa2,maxa3,us2[20],us3[20],in[4]={U(0),U(1),0,0},q[10][4][20];
int_T  row_optim,row_optimal_action_allc[20],columb_optimal_action,row_optimal_action;
real_T optimal_action_c,optimal_action_allc[20],optimal_action; 
int_T i,j,k;
int_T m1234=n*n*n*tiling, m234=n*n*tiling, m34=n*tiling, m4=tiling;

    UNUSED_ARG(tid); 
for (i=0;i<tiling;i++)
	for (j=0;j<n;j++)
	   for (k=0;k<DIM;k++)
	        q[j][k][i] = (j)*incr[k]+(i)*off[k]+mins[k];
    /* y=Cx+Du */
/*minn = q[0][2][0]; */
minn = 0; maxx = 15;
inc = (maxx - minn) / 10;
mina2 = U(2) - inc; mina3 = U(3) - inc;  
maxa2 = U(2) + inc; maxa3 = U(3) + inc;
if (mina2 < minn)
   mina2 = minn;
   if (mina3 < minn)
	   mina3 = minn;
      if (maxa2 > maxx)
		  maxa2 = maxx;
         if (maxa3 > maxx)
		     maxa3 = maxx;  
for (ifloat2=mina2;ifloat2<maxa2;ifloat2+=(maxa2-mina2)/(RES-1))
{   us2[jp2] = ifloat2;
    jp2++;}
us2[RES-1] = maxa2;
for (ifloat3=mina3;ifloat3<maxa3;ifloat3+=(maxa3-mina3)/(RES-1))
{   us3[jp3] = ifloat3;
    jp3++;}
us3[RES-1] = maxa3;

for (i=0;i<tiling;i++)
	for (k=0;k<2;k++)
	{ j = n-1;
     while (j>0 && U(k)<q[j][k][i])
	 {	    j = j - 1;  }
	 a_optim[i][k] = j;
	}

for (ij=0;ij<RES;ij++)     
{   DIMc = 4; 
   for (ii=0;ii<RES;ii++) 
   {   in[2] = us2[ii];  
       in[3] = us3[ij];
/* function a = activate(q,x) */
for (i=0;i<tiling;i++)
   	for (k=2;k<DIMc;k++)
	{ j = n-1;
     while (j>0 && in[k]<q[j][k][i])
	 {	    j = j - 1;  }
	 a_optim[i][k] = j;
	} 
DIMc = 3; 	
y_optim = 0;
for (i=0;i<tiling;i++)
{   k0 = a_optim[i][0]; k1 = a_optim[i][1]; k2 = a_optim[i][2]; k3 = a_optim[i][3];
    Locate_Q_optim = k0*m1234+k1*m234+k2*m34+k3*m4+i;
	y_optim+=x[Locate_Q_optim];
}
Q_table_optim[ii][ij] = y_optim; 
   } 
}     
/* Search for optimal action at Q_table_optim (need more action!) */
   for (j=0;j<RES;j++)
   {   optimal_action_c = 1000000000; 
	   for (i=0;i<RES;i++)
	      if (optimal_action_c > Q_table_optim[i][j])
	   {      optimal_action_c = Q_table_optim[i][j];
			  row_optim = i;
	   }
	   optimal_action_allc[j] = optimal_action_c; 
	   row_optimal_action_allc[j] = row_optim;
   }
optimal_action = 1000000000;
for (i=0;i<RES;i++)
     if (optimal_action > optimal_action_allc[i])
	 {  optimal_action = optimal_action_allc[i];
	    columb_optimal_action = i;
        row_optimal_action = row_optimal_action_allc[i];
	 }

y[0] = us2[row_optimal_action];               
y[1] = us3[columb_optimal_action];            
if (mina2 > maxx)
   y[0] = 15;
if (maxa2 < minn)
    y[0] = 0; 
if (mina3 > maxx)
    y[1] = 15;
if (maxa3 < minn)
    y[1] = 0; 

}

#define MDL_UPDATE
/* Function: mdlUpdate ======================================================
 * Abstract:
 *    x(t+1) = Ax(t) + Bu(t)
 */
static void mdlUpdate(SimStruct *S, int_T tid)
{
    real_T            *x       = ssGetRealDiscStates(S);
    InputRealPtrsType uPtrs    = ssGetInputPortRealSignalPtrs(S,0);
int_T  tiling=20,n=10,DIM=4,RES=20;
int_T  Size_state_Q=200000, Size_state_extra_Q=200080;
real_T alpha=0.7,gamma=0.95;
int_T  k0,k1,k2,k3,Locate_Q,pa=Size_state_Q, ax[20][4], h0,h1,h2,h3,Locate_Q_past,la=Size_state_Q;
real_T a[20][4],y,Q_sp_ap,tar,cur,diff,q[10][4][20];
int_T i,j,k;
int_T m1234=n*n*n*tiling, m234=n*n*tiling, m34=n*tiling, m4=tiling;

    UNUSED_ARG(tid); /* not used in single tasking mode */
for (i=0;i<tiling;i++)
	for (j=0;j<n;j++)
	   for (k=0;k<DIM;k++)
	        q[j][k][i] = (j)*incr[k]+(i)*off[k]+mins[k];
/* x(t+1) = Ax(t)+Bu(t) */
/* function a = activate(q,x) */
/* This command computes the activation for a CMAC neural network with particular inputs. */
for (i=0;i<tiling;i++)
	for (k=0;k<4;k++)
	{ j = n-1;
     while (j>0 && U(k)<q[j][k][i])
	 {	    j = j - 1;  }
	 a[i][k] = j;
	}
/*  */
/* This command computes the output of a CMAC neural network. */
for (i=0;i<tiling;i++)
{   k0 = a[i][0]; k1 = a[i][1]; k2 = a[i][2]; k3 = a[i][3];
    Locate_Q = k0*m1234+k1*m234+k2*m34+k3*m4+i;
	y+=x[Locate_Q];
}
Q_sp_ap = y; 
/* function Q = learnq(tar,cur,activ,Q,alpha) */
/* This command train the CMAC network using the TD-error. */
tar = gamma * Q_sp_ap + U(4);                     
cur = x[Size_state_extra_Q];                      
diff = (tar - cur) * alpha / tiling;              /* Q_sp_ap : Q(s',a')  /U(4) : Reward */ 
/* This part convert a part of state vector to matrix of activation location. */
for (i=0;i<DIM;i++)
    for (j=0;j<tiling;j++)
	    ax[j][i] = x[pa+j+i*tiling]; 
for (i=0;i<tiling;i++)
{   h0 = ax[i][0]; h1 = ax[i][1]; h2 = ax[i][2]; h3 = ax[i][3];
    Locate_Q_past = h0*m1234+h1*m234+h2*m34+h3*m4+i;
    x[Locate_Q_past] = x[Locate_Q_past] + diff;
}
x[Size_state_extra_Q] = Q_sp_ap; 
/**/
/* This part is for convert matrix 'a' to vector. */
for (i=0;i<DIM;i++)
   for (j=0;j<tiling;j++)
   {	x[la+j+i*tiling] = a[j][i]; }
/*********************************************************/

}
/* Function: mdlTerminate =====================================================
 * Abstract:
 */
static void mdlTerminate(SimStruct *S)
{
    real_T            *x    = ssGetRealDiscStates(S);
	int_T io;
	FILE *fptr;
    fptr = fopen("C:\MATLAB6p5\work\Mother.txt","w");
	for (io=0;io<200081;io++)
	{fprintf(fptr,"%f \n",x[io]);}
	fclose(fptr);
	
/* comment:  add more efficient search */
}

#ifdef  MATLAB_MEX_FILE    
#include "simulink.c"      
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
