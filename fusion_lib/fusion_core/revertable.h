#ifndef __REVERTABLE__
#define __REVERTABLE__

namespace fusion
{

template <class T>
class Revertable
{
public:
  Revertable() = default;
  Revertable(T other);
  void update(T other);
  void revert();
  T value() const;

private:
  T current;
  T last;
};

template <class T>
Revertable<T>::Revertable(T other)
{
  current = other;
}

template <class T>
void Revertable<T>::update(T other)
{
  last = current;
  current = other;
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