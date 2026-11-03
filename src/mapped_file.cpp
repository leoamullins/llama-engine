#include "mapped_file.hpp"
#include <fcntl.h>    // copy
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close
#include <stdexcept>
#include <utility>    // exchange

MappedFile::MappedFile(const std::string &path)
{
  // open with read_only
  int fd = open(path.c_str(), O_RDONLY);

  // ensure file descriptor is valid
  if (fd < 0)
  {
    throw(std::invalid_argument("invalid_argument"));
  }

  struct stat st;
  if (fstat(fd, &st) < 0)
  {
    close(fd);
    throw(std::invalid_argument("invalid_argument"));
  }
  size_ = st.st_size;

  void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  // ensure the mapping succeeded (also fails for an empty file)
  if (p == MAP_FAILED)
  {
    throw(std::runtime_error("failed to map file: " + path));
  }

  data_ = static_cast<const std::byte *>(p);
}

MappedFile::MappedFile(MappedFile &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0))
{
}

MappedFile::~MappedFile()
{
  if (data_)
  {
    munmap(const_cast<std::byte *>(data_), size_);
  }
}
