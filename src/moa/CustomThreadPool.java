package moa;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class CustomThreadPool {
    public long[] threadIDs;
    ForkJoinPool pool;
    int size_;

    public CustomThreadPool(int size){
        pool = new ForkJoinPool(size);
        threadIDs = new long[size];
        size_ = size;

    }
    public void getThreadIDs() throws ExecutionException, InterruptedException {

        pool.submit(() -> IntStream.range(0, size_).parallel().forEach(i -> {
            try {
                train(i);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        })).get();;


    }

    private void train(int i) throws InterruptedException {
        threadIDs[i] = Thread.currentThread().getId();
        Thread.sleep(1000);


    }

}
