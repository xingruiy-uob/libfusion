#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <mutex>

template <class T>
class SafeQueue
{
public:
    void push(T &val);
    void push_safe(T &val);
    void pop_safe();
    T front_safe();
    size_t size_safe();

private:
    std::mutex _lock;
    std::queue<T> _queue;
};

template <class T>
void SafeQueue<T>::push(T &val)
{
    _queue.push(val);
}

template <class T>
void SafeQueue<T>::push_safe(T &val)
{
    std::unique_lock<std::mutex> lock(_lock);
    _queue.push(val);
}

template <class T>
void SafeQueue<T>::pop_safe()
{
    std::unique_lock<std::mutex> lock(_lock);
    _queue.pop();
}

template <class T>
T SafeQueue<T>::front_safe()
{
    std::unique_lock<std::mutex> lock(_lock);
    _queue.front();
}

template <class T>
size_t SafeQueue<T>::size_safe()
{
    std::unique_lock<std::mutex> lock(_lock);
    return _queue.size();
}

#endif