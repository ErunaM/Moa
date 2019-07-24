package moa.classifiers;

import java.util.concurrent.ForkJoinPool;

public interface InterfaceMT {

     void ReceivePool(ForkJoinPool pool);

     void trainingHasEnded();

     int getCores();


}
