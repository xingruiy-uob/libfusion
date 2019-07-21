#ifndef FUSION_DATA_STRUCT_DEVICE_ARRAY_H
#define FUSION_DATA_STRUCT_DEVICE_ARRAY_H

#include <atomic>
#include <vector>
#include "macros.h"
#include "utils/safe_call.h"

namespace fusion
{

template <class T>
struct PtrSz
{
    FUSION_DEVICE inline T &operator[](int x) const;
    FUSION_DEVICE inline operator T *() const;

    T *data;
    size_t size;
};

template <class T>
FUSION_DEVICE inline T &PtrSz<T>::operator[](int x) const
{
    return data[x];
}

template <class T>
FUSION_DEVICE inline PtrSz<T>::operator T *() const
{
    return data;
}

template <class T>
struct PtrStep
{
    FUSION_DEVICE inline T *ptr(int y = 0) const;

    T *data;
    size_t step;
};

template <class T>
FUSION_DEVICE inline T *PtrStep<T>::ptr(int y) const
{
    return (T *)((char *)data + y * step);
}

template <class T>
struct PtrStepSz
{
    FUSION_DEVICE inline T *ptr(int y = 0) const;

    T *data;
    int cols;
    int rows;
    size_t step;
};

template <class T>
FUSION_DEVICE inline T *PtrStepSz<T>::ptr(int y) const
{
    return (T *)((char *)data + y * step);
}

template <class T>
struct DeviceArray
{
    FUSION_HOST inline DeviceArray();
    FUSION_HOST inline ~DeviceArray();
    FUSION_HOST inline DeviceArray(const size_t alloc_size);
    FUSION_HOST inline DeviceArray(const DeviceArray &array);
    FUSION_HOST inline DeviceArray<T> &operator=(DeviceArray<T> array);

    template <class U>
    FUSION_HOST inline friend void swap(DeviceArray<U> &a, DeviceArray<U> &b);

    FUSION_HOST inline void create(const size_t alloc_size);
    FUSION_HOST inline void upload(const void *const host_data);
    FUSION_HOST inline void download(void *const host_data) const;
    FUSION_HOST inline void set_to(const int val);
    FUSION_HOST inline void release();
    FUSION_HOST inline bool empty() const;
    FUSION_HOST inline void copy_to(DeviceArray<T> &other) const;

    FUSION_HOST inline operator T *() const;
    FUSION_HOST inline operator PtrSz<T>() const;

    void *data;
    size_t size;
    std::atomic<int> *ref_counter;
};

template <class T>
FUSION_HOST inline DeviceArray<T>::DeviceArray()
    : data(0), ref_counter(0), size(0) {}

template <class T>
FUSION_HOST inline DeviceArray<T>::DeviceArray(const size_t alloc_size)
    : data(0), ref_counter(0), size(alloc_size)
{
    create(alloc_size);
}

template <class T>
FUSION_HOST inline DeviceArray<T>::DeviceArray(const DeviceArray<T> &array)
    : data(array.data), size(array.size), ref_counter(array.ref_counter)
{
    if (!array.empty())
        array.ref_counter->fetch_add(1);
}

template <class T>
FUSION_HOST inline void swap(DeviceArray<T> &a, DeviceArray<T> &b)
{
    if (&a != &b)
    {
        std::swap(a.data, b.data);
        std::swap(a.size, b.size);
        std::swap(a.ref_counter, b.ref_counter);
    }
}

template <class T>
FUSION_HOST inline DeviceArray<T> &DeviceArray<T>::operator=(DeviceArray<T> array)
{
    if (this != &array)
    {
        swap(*this, array);
    }
    return *this;
}

template <class T>
FUSION_HOST inline DeviceArray<T>::~DeviceArray()
{
    release();
}

template <class T>
FUSION_HOST inline bool DeviceArray<T>::empty() const
{
    return data == NULL;
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::create(const size_t alloc_size)
{
    if (!empty())
        release();

    size = alloc_size;
    ref_counter = new std::atomic<int>(1);
    safe_call(cudaMalloc(&data, sizeof(T) * alloc_size));
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::upload(const void *const host_data)
{
    upload(host_data, size);
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::download(void *const host_data) const
{
    safe_call(cudaMemcpy(host_data, data, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::set_to(const int val)
{
    safe_call(cudaMemset(data, val, sizeof(T) * size));
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::release()
{
    if (ref_counter && ref_counter->fetch_sub(1) == 1)
    {
        delete ref_counter;

        if (data)
            safe_call(cudaFree(data));
    }

    size = 0;
    data = 0;
    ref_counter = 0;
}

template <class T>
FUSION_HOST inline void DeviceArray<T>::copy_to(DeviceArray<T> &array) const
{
    if (empty())
    {
        array.release();
        return;
    }

    array.create(size);
    safe_call(cudaMemcpy(array.data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template <class T>
FUSION_HOST inline DeviceArray<T>::operator T *() const
{
    return (T *)data;
}

template <class T>
FUSION_HOST inline DeviceArray<T>::operator PtrSz<T>() const
{
    PtrSz<T> device_ptr;
    device_ptr.data = (T *)data;
    device_ptr.size = size;
    return device_ptr;
}

template <class T>
struct DeviceArray2D
{
    FUSION_HOST inline DeviceArray2D();
    FUSION_HOST inline ~DeviceArray2D();
    FUSION_HOST inline DeviceArray2D(const int cols, const int rows);
    FUSION_HOST inline DeviceArray2D(const DeviceArray2D<T> &array);

    template <class U>
    FUSION_HOST inline friend void swap(DeviceArray2D<U> &a, DeviceArray2D<U> &b);
    FUSION_HOST inline DeviceArray2D<T> &operator=(DeviceArray2D<T> array);

    FUSION_HOST inline void create(const int cols, const int rows);
    FUSION_HOST inline void upload(const void *const host_data, const size_t host_step);
    FUSION_HOST inline void download(void *const host_data, const size_t host_step) const;
    FUSION_HOST inline void set_to(const T val);
    FUSION_HOST inline void release();
    FUSION_HOST inline bool empty() const;
    FUSION_HOST inline void copy_to(DeviceArray2D<T> &array) const;

    FUSION_HOST inline operator T *() const;
    FUSION_HOST inline operator PtrStep<T>() const;
    FUSION_HOST inline operator PtrStepSz<T>() const;

    int cols;
    int rows;
    void *data;
    size_t step;
    std::atomic<int> *ref_counter;
};

template <class T>
FUSION_HOST inline DeviceArray2D<T>::DeviceArray2D()
    : data(0), ref_counter(0), step(0), cols(0), rows(0)
{
}

template <class T>
FUSION_HOST inline DeviceArray2D<T>::DeviceArray2D(const int cols, const int rows)
    : data(0), ref_counter(0), step(0), cols(cols), rows(rows)
{
    create(cols, rows);
}

template <class T>
FUSION_HOST inline DeviceArray2D<T>::DeviceArray2D(const DeviceArray2D &array)
    : data(array.data), step(array.step), cols(array.cols),
      rows(array.rows), ref_counter(array.ref_counter)
{
    if (!array.empty())
        array.ref_counter->fetch_add(1);
}

template <class T>
FUSION_HOST inline DeviceArray2D<T>::~DeviceArray2D()
{
    release();
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::create(const int cols, const int rows)
{
    if (cols > 0 && rows > 0)
    {
        if (!empty())
            release();

        safe_call(cudaMallocPitch(&data, &step, sizeof(T) * cols, rows));

        this->cols = cols;
        this->rows = rows;
        ref_counter = new std::atomic<int>(1);
    }
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::upload(const void *const host_data, const size_t host_step)
{
    safe_call(cudaMemcpy2D(data, step, host_data, host_step, sizeof(T) * cols, rows, cudaMemcpyHostToDevice));
}

template <class T>
FUSION_HOST inline bool DeviceArray2D<T>::empty() const
{
    return data == NULL;
}

template <class T>
FUSION_HOST inline void swap(DeviceArray2D<T> &a, DeviceArray2D<T> &b)
{
    if (&a != &b)
    {
        std::swap(a.data, b.data);
        std::swap(a.cols, b.cols);
        std::swap(a.rows, b.rows);
        std::swap(a.step, b.step);
        std::swap(a.ref_counter, b.ref_counter);
    }
}

template <class T>
FUSION_HOST inline DeviceArray2D<T> &DeviceArray2D<T>::operator=(DeviceArray2D<T> array)
{
    if (this != &array)
    {
        swap(*this, array);
    }
    return *this;
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::set_to(const T val)
{
    safe_call(cudaMemset2D(data, step, val, sizeof(T) * cols, rows));
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::download(void *const host_data, const size_t host_step) const
{
    if (empty())
        return;

    safe_call(cudaMemcpy2D(
        host_data, host_step,
        data, step,
        sizeof(T) * cols, rows,
        cudaMemcpyDeviceToHost));
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::release()
{
    if (ref_counter && ref_counter->fetch_sub(1) == 1)
    {
        delete ref_counter;
        if (data)
            safe_call(cudaFree(data));
    }
    cols = rows = step = 0;
    data = ref_counter = 0;
}

template <class T>
FUSION_HOST inline void DeviceArray2D<T>::copy_to(DeviceArray2D<T> &array) const
{
    if (empty())
        array.release();

    array.create(cols, rows);
    safe_call(cudaMemcpy2D(
        array.data, array.step,
        data, step,
        sizeof(T) * cols, rows,
        cudaMemcpyDeviceToDevice));
}

template <class T>
FUSION_HOST inline DeviceArray2D<T>::operator PtrStep<T>() const
{
    PtrStep<T> device_ptr;
    device_ptr.data = (T *)data;
    device_ptr.step = step;
    return device_ptr;
}

template <class T>
FUSION_HOST inline DeviceArray2D<T>::operator PtrStepSz<T>() const
{
    PtrStepSz<T> device_ptr;
    device_ptr.data = (T *)data;
    device_ptr.cols = cols;
    device_ptr.rows = rows;
    device_ptr.step = step;
    return device_ptr;
}

} // namespace fusion

#endif