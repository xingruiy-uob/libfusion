#ifndef THREAD_QUEUE_H
#define THREAD_QUEUE_H

#include <queue>
#include <mutex>

template <class T>
class ThreadQueue
{
public:
    void push(const T &val);
    void push_sync(const T &val);
    void pop_sync();
    T front_sync();
    size_t size_sync();

private:
    std::queue<T> _queue;
    std::mutex unique_lock;
};

template <class T>
void ThreadQueue<T>::push(const T &val)
{
    _queue.push(val);
}

template <class T>
void ThreadQueue<T>::push_sync(const T &val)
{
    std::lock_guard<std::mutex> lock(unique_lock);
    _queue.push(val);
}

template <class T>
void ThreadQueue<T>::pop_sync()
{
    std::lock_guard<std::mutex> lock(unique_lock);
    _queue.pop();
}

template <class T>
T ThreadQueue<T>::front_sync()
{
    std::lock_guard<std::mutex> lock(unique_lock);
    return _queue.front();
}

template <class T>
size_t ThreadQueue<T>::size_sync()
{
    std::lock_guard<std::mutex> lock(unique_lock);
    return _queue.size();
}

#endif