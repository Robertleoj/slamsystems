#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template <typename T>
class ThreadQueue {
   private:
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::optional<int> max_size;

    void delete_old() {
        if (!max_size.has_value()) {
            return;
        }

        while (queue.size() > max_size.value()) {
            queue.pop();
        }
    }

   public:
    ThreadQueue(
        std::optional<int> max_size = std::nullopt
    )
        : max_size(max_size) {}

    void push(
        T value
    ) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(std::move(value));
        delete_old();
        cv.notify_one();
    }

    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] {
            return !queue.empty();
        });
        auto value = std::move(queue.front());
        queue.pop();

        return value;
    }

    std::optional<T> try_pop() {
        std::scoped_lock l(mtx);

        if (queue.empty()) {
            return std::nullopt;
        }

        auto value = std::move(queue.front());
        queue.pop();

        return value;
    }

    int size() const {
        std::scoped_lock(mtx);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};
