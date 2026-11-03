#pragma once
#include <cstddef>
#include <string>

class MappedFile
{
public:
  explicit MappedFile(const std::string &path);
  ~MappedFile();

  MappedFile(const MappedFile &) = delete;            // no copy constructing
  MappedFile &operator=(const MappedFile &) = delete; // no copy assinging
  MappedFile(MappedFile &&other) noexcept;

private:
  const std::byte *data_ = nullptr;
  std::size_t size_ = 0;
};
