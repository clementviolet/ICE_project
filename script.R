# Age or stage-sctructure models

rm(list¼ls()) # Clear workspace
Leslie.matrix <- matrix(
c(0.8, 1.2, 1.0, 0,
0.8, 0.0, 0.0, 0,
0.0, 0.4, 0.0, 0,
0.0, 0.0, 0.25,0),4,4, byrow = TRUE)

Eigen.data <- eigen( Leslie.matrix)
Lambda <-Eigen.data$values[1] # Get first eigenvalue
Maxgen <-12
# Number of generations simulation runs
n <- c(1,0,0,0) # Initial population
# Preassign matrix to hold cohort number and total population size
Pop <- matrix(0,Maxgen,5)
Pop[1,] <- c(n[1:4], sum(n)) # Store initial population
# Preassign storage for observed lambda
Obs.lambda <- matrix(0,Maxgen,5)
for ( Igen in 2:Maxgen) # Iterate over generations
{
n <-Leslie.matrix%*%n # Apply matrix multiplication
Pop[Igen,1:4]<-n[1:4] # Store cohorts
Pop[Igen,5] <-sum(n) # Store total population size
Obs.lambda[Igen,] <-Pop[Igen,]/Pop[Igen-1,]
# Store observed lambda
} # End of Igen loop
# Print out observed lambda in last generation and ratio
print(c(Obs.lambda[Maxgen], Obs.lambda[Maxgen]/Lambda))
par(mfrow¼c(2,2)) # Make 2x2 layout of plots
Generation <-seq(from=1, to=Maxgen) # Vector of generation number
# Plot population and cohort trajectories
ymin <-min(Pop); ymax <-max(
Pop) # get minimum and maximum pop sizes
plot( Generation, Pop[,1], type=’l’,ylim¼c(ymin,ymax),
ylab¼’Population and cohort sizes’) # Cohort 1
for( i in 2:4) {lines(Generation, Pop[,i]) } # Cohorts 24
lines(Generation, Pop[,5], lty¼2) # Total population
# Plot log of population and cohort trajectories
# Log zero is undefined so remove these
x <-matrix(Pop,length(Pop),1) # Convert to one dimensional matrix
ymin <- min(log(x[x!=0])) # minimum log value
ymax <max(
log(Pop)) # get minimum and maximum pop sizes
plot( Generation, log(Pop[,1]), type¼’l’, ylim¼c(ymin,ymax),
ylab¼’log Sizes’)
for(i in 2:4) {lines(Generation, log(Pop[,i]))}
lines(Generation, log(Pop[,5]), lty¼2) # Total population
# Plot Observed lambdas
plot(Generation, Obs.lambda[,1], type¼’l’, ylab¼’Lambda’)
for( i in 2:4) {lines(Generation, Obs.lambda[,i])}
lines(Generation, Obs.lambda[,5], lty¼2) # Total population
# Plot observed r
plot(Generation, log(Obs.lambda[,1]), type¼’l’, ylab¼’r’)
for( i in 2:4) {lines(Generation, log(Obs.lambda[,i]))}
lines(Generation, log(Obs.lambda[,5]), lty¼2)
# Total population

#### Adding density dependence

rm(list=ls()) # Clear workspace
par(mfrow=c(3,2)) # Divide page into 6 panels
BH.FUNCTION <-function(n,c1,c2) {c1/(1+c2*n)}
RICKER.FUNCTION <-function(n, ALPHA, BETA) {ALPHA*exp(-BETA*n)}
################### MAIN PROGRAM ###################
########## Beverton Holt function ##########
c1 <-100;
c2 <-2*10^-3
# BH parameters
# Plot N(t) on t
Maxgen <-20;
N.t <-matrix(0,Maxgen); N.t[1] <-1

for (i in 2:Maxgen){
    N.t[i] <-N.t[i-1]*BH.FUNCTION(N.t[i-1],c1,c2)}
plot(seq(from=1, to=Maxgen), N.t, xlab = 'Generation, t', ylab='N(t)',type='l')
# Plot N(t+1) on N(t)
MaxN <-5000;
N.t <-matrix(seq(from=1, to=MaxN))
N.tplus1 <-N.t*apply(N.t,1,BH.FUNCTION, c1,c2)
plot(N.t, N.tplus1, xlab = 'N(t)', ylab='N(t+1)', type='l')
########## Ricker function ##########
ALPHA <-c(6, 60); BETA <-.0005
# Parameter values
# Plot N(t) on t for 2 values of ALPHA
Maxgen <-40
for (j in 1:2){
N.t <-matrix(0,Maxgen,1); N.t[1] <-1
for ( i in 2:Maxgen){
    N.t[i]<-N.t[i-1]*RICKER.FUNCTION(N.t[i-1],ALPHA[j], BETA)
    }
plot(seq(from=1, to=Maxgen), N.t, xlab = 'Generation, t', ylab
='N(t)', type='l')
# Plot N(t+1) on N(t)
MaxN <-10000;
N.t <-matrix(seq(from=1, to=MaxN))
N.tplus1 <-N.t*apply(N.t, 1, RICKER.FUNCTION, ALPHA[j],BETA)
plot(N.t, N.tplus1, xlab='N(t)', ylab='N(t+1)', type='l')
lines(N.t, N.t)
} # End of j loop