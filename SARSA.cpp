/*
  Temporal difference (TD) learning methods
  SARSA(Lambda)
  With replacing traces
  Action selection : e-greedy policy = f(Q values)

  Compilation instructions:

  gcc -o sarsa_lambda -lm -g sarsa_lambda.c

  then run ./sarsa_lambda
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define min(x, y)               ((x <= y) ? x : y)
#define max(x, y)               ((x >= y) ? x : y)

#define Alpha 0.1             /* learning rate */
#define Epsilon 0.01          /* percentage of exploration */ 

/* size of the maze */
#define ROW 6
#define COL 7

#define A 4           /* number of actions */
#define S 16          /* number of states */

/* Goal */
#define G_i 3
#define G_j 5

#define EPISODES 150    /* in hundreds */
#define MaxSteps 1000   /* maximum number of tries per episode */

/* environment :  1=obstacle   0=no obstacle */
int ENV[ 7][ 6] = {{1,1,1,1,1,1},
                   {1,0,0,1,0,1},
                   {1,0,0,0,0,1},
                   {1,0,1,1,0,1},
                   {1,1,1,1,0,1},
                   {1,1,1,0,0,1},
                   {1,1,1,1,1,1}};

/* predeclarations */
int select_action(float Q[S][A], int s);
void InitQfunc(float Q[S][A]);
int InitState(int *x, int *y, float e[S][A]);
int GetState(int x, int y);
int reward(int x, int y, int a);
int NextState(int *x,int *y,int a);
void UpdateQfunc(int s, int a, int r, float Q[S][A], int next_s, int next_a, float e[S][A],float Gamma,float Lambda,float GtotheL);
void end();

void main(int argc, char *argv[])
{
 float Q[S][A];         /* Q state-action values */
 float e[S][A];         /* eligibility traces */

 float Gamma, Lambda, GtotheL;

 int a_t;  
 int s;
 int x,y;
 int next_s, next_a;
 int r;
 int sum_r,avg_r;

 int i,j,cnt;
 int sum_steps,sum_failures;

 Gamma = 0.9;      /* discount factor */
 Lambda = 0.9;     /* lambda parameter in SARSA(lambda) */
 GtotheL = 0.9095; /* Gamma to the Lambda */

 srand48(123456789);     
 InitQfunc(Q);           
 sum_steps = 0;
 sum_failures = 0; 
 for(i=0;i<(EPISODES);i++) {
  avg_r = 0.0;
	for(j=0;j<100;j++) {
  s=InitState(&x,&y,e);
  cnt=0;
  sum_r=0;

  /* repeat for a certain maximum number of steps or until goal is reached */
  while(((x != G_i) || (y != G_j)) && (cnt < MaxSteps)) {
   a_t = select_action(Q,s);         /* select an action */
   next_s = NextState(&x,&y,a_t);    /* apply action */
   r = reward(x,y,a_t);              /* receive external reinforcement */
   next_a = select_action(Q,next_s); /* select next action */

   /* update Q values */
   UpdateQfunc(s,a_t,r,Q,next_s,next_a,e,Gamma,Lambda,GtotheL);

   /* update current state */ 
   s = next_s;

   cnt++;
   sum_r +=r;
  }
   avg_r +=sum_r;
	}

	/* print the average of sum of gained reward */ 
  printf("\n%f",avg_r/100.0);
 }
 end(Q);
}

int InitState(int *x,int *y,float e[S][A])
{
 int i,j,s;
 
 
 *x = lrand48()%2 + 1;
 *y = lrand48()%3 + 1;

 s = GetState(*x,*y);

 for(i=0;i<S;i++)
  for(j=0;j<A;j++)
   e[i][j] = 0.0;

 return(s);
}
 
int select_action(float Q[S][A],int s)
{
 int i;
 int action; 
 int a_qmax; 

 a_qmax =0;
 /* find the action with maximum Q value, given a certain state s */
 for(i=1;i<A;i++)
  if(Q[s][i] > Q[s][a_qmax])
   a_qmax = i;

 action = a_qmax;

 /* chose the action with maximum Q value with probability 1-epsilon % 
    else, chose a random action */
  if(drand48() < Epsilon)
  action=lrand48()%A;
 return(action);
}

void InitQfunc(float Q[S][A])
{
 int s,a;

 for(s=0;s<S;s++)
  for(a=0;a<A;a++)
   Q[s][a] = 0.0;
}

/* return a negative reinforcement if goal is not reached yet
   else return zero as reinforcement */
int reward(int x, int y, int a)
{
 if((x == G_i) && (y == G_j))
  return(0);
 else return(-1); 
}

int GetState(int x, int y)
{
 int s = 0;

 if(ENV[y-1][x]) s++;
 if(ENV[y][x-1]) s +=2;
 if(ENV[y][x+1]) s +=4;
 if(ENV[y+1][x]) s +=8;
  
 return(s);
}

int NextState(int *x, int *y, int a)
{
  int i,j;
  int s;

  switch(a) {
   case 0: if(!ENV[*y-1][*x]) 
            *y -= 1;
           break;
   case 1: if(!ENV[*y][*x+1]) 
            *x += 1;
           break;
   case 2: if(!ENV[*y+1][*x]) 
            *y += 1;
           break;
   case 3: if(!ENV[*y][*x-1]) 
            *x -= 1;
           break;
  }
  s = GetState(*x,*y);
  return(s); 
}

void UpdateQfunc(int s, int a, int r, float Q[S][A], int next_s, int next_a, float e[S][A],float Gamma,float Lambda,float GtotheL)
{
 int i,j;
 float TDerr;
 
 
 /* compute TD error */
 if(next_s != -1)   
  TDerr = (float)r+Gamma*Q[next_s][next_a] - Q[s][a];
 else TDerr = (float)r - Q[s][a];

 /* replacing traces:  */
 e[s][a] = 1; 

 for(i=0;i<S;i++)
  for(j=0;j<A;j++) {

   /* update Q values */
   Q[i][j] += Alpha*TDerr*e[i][j];

   e[i][j] = (float)(GtotheL*e[i][j]);

  }
}

void end(float Q[S][A])
{
 int i,j;
 int a_qmax;

 printf("\nLearned policy:\n");

 for(i=0;i<S;i++) {
  a_qmax = 0;
  for(j=1;j<A;j++)
   if(Q[i][j] > Q[i][a_qmax])
    a_qmax = j;
  printf("\nA[%d] = %d",i,a_qmax);
 }
}
