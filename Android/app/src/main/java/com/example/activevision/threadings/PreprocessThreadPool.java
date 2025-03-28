package com.example.activevision.threadings;

import android.util.Log;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;


/**
 * It is responsible for managing a thread pool that performs various preprocessing tasks
 * such as image processing or data loading.
 * Author: Zhiyuan Lu
 * Date: 21/03/2025
 */

public class PreprocessThreadPool {
    private static volatile PreprocessThreadPool instance = null;

    private final ThreadPoolExecutor threadPoolExecutor;

    private PreprocessThreadPool() {
        threadPoolExecutor = new ThreadPoolExecutor(
                2,
                4,
                1L,
                TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(200),
                new NamingThreadFactory("PreprocessThread"),
                new ThreadPoolExecutor.DiscardPolicy()
        );
        printThreadPoolStatus(threadPoolExecutor);
    }

    // Singleton instance
    public static PreprocessThreadPool getInstance() {
        if (instance == null) {
            synchronized (PreprocessThreadPool.class) {
                if (instance == null) {
                    instance = new PreprocessThreadPool();
                }
            }
        }
        return instance;
    }

    // Method to submit tasks to the thread pool
    public void submitTask(Runnable task) {
        threadPoolExecutor.submit(task);
    }

    // Method to shut down the thread pool
    public void shutdown() {
        threadPoolExecutor.shutdownNow();
    }

    public static void printThreadPoolStatus(ThreadPoolExecutor threadPool) {
        ScheduledExecutorService scheduledExecutorService = new ScheduledThreadPoolExecutor(
                1, new NamingThreadFactory("ThreadPoolStatusPrinter")
        );
        scheduledExecutorService.scheduleWithFixedDelay(() -> {
            Log.i("PreprocessThreadPool", "=========================");
            Log.i("PreprocessThreadPool", "ThreadPool Size: [" + threadPool.getPoolSize() + "]");
            Log.i("PreprocessThreadPool", "Active Threads: " + threadPool.getActiveCount());
            Log.i("PreprocessThreadPool", "Number of Tasks Completed: " + threadPool.getCompletedTaskCount());
            Log.i("PreprocessThreadPool", "Number of Tasks in Queue: " + threadPool.getQueue().size());
            Log.i("PreprocessThreadPool", "=========================");
        }, 0, 1, TimeUnit.SECONDS);
    }
}
