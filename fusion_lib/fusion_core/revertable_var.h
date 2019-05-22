#ifndef __REVERTABLE_VAR__
#define __REVERTABLE_VAR__

namespace fusion
{

template <class T>
class Revertable
{
public:
  Revertable() = default;
  Revertable(const T &);
  void operator=(const T &);
  void update(const T &);
  void revert();
  T value() const;

private:
  T current;
  T last;
};

template <class T>
Revertable<T>::Revertable(const T &val)
{
  current = val;
}

template <class T>
void Revertable<T>::operator=(const T &val)
{
  update(val);
}

template <class T>
void Revertable<T>::update(const T &val)
{
  last = current;
  current = val;
}

template <class T>
void Revertable<T>::revert()
{
  current = last;
}

template <class T>
T Revertable<T>::value() const
{
  return current;
}

} // namespace fusion

#endif