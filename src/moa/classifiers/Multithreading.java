package moa.classifiers;

import java.util.concurrent.ForkJoinPool;

public interface Multithreading {

     void ReceivePool(ForkJoinPool pool);

     void trainingHasEnded();

     int getCores();

     double getCpuTime();


}
