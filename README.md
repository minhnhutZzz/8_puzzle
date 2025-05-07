# Giới thiệu bài toán 8_puzzle
8-Puzzle Solver là một đồ án phần mềm giải bài toán 8-Puzzle (trò chơi xếp ô số 3x3) được phát triển bằng Python, sử dụng Pygame cho giao diện người dùng và Plotly để trực quan hóa hiệu suất. Chương trình tích hợp hơn 20 thuật toán tìm kiếm để giải bài toán và so sánh hiệu suất.
# 1. Mục tiêu
Mục tiêu cốt lõi của dự án là xây dựng một chương trình toàn diện và linh hoạt để giải quyết bài toán 8-puzzle – một bài toán kinh điển và nền tảng trong lĩnh vực trí tuệ nhân tạo cũng như khoa học máy tính – thông qua việc tích hợp nhiều thuật toán đa dạng và tiên tiến, từ các phương pháp tìm kiếm truyền thống đến các kỹ thuật học tăng cường hiện đại. Dự án không chỉ dừng lại ở việc tạo ra một công cụ đơn thuần để tìm lời giải cho bài toán, mà còn đặt trọng tâm vào việc thiết kế một nền tảng mạnh mẽ, hỗ trợ nghiên cứu chuyên sâu, học tập thực tiễn, và khám phá các cách tiếp cận khác nhau trong việc giải quyết các vấn đề phức tạp của trí tuệ nhân tạo, từ đó mang lại giá trị giáo dục và thực tiễn cho người dùng.
# 2. Nội dung
## 2.1 Nhóm thuật toán tìm kiếm không có thông tin (Uninformed Search Algorithms)
Các thành phần chính của bài toán tìm kiếm và giải pháp
+ Trạng thái ban đầu
  - Một lưới 3x3 với 8 số từ 1 đến 8 và một ô trống (0), đại diện cho trạng thái khởi đầu của bài toán.
+ Trạng thái mục tiêu
  - Lưới 3x3 với thứ tự số từ 1 đến 8 và ô trống ở vị trí cuối cùng ([[1 2 3], [4 5 6], [7 8 0]]).
+ Không gian trạng thái
  - Tập hợp tất cả các cấu hình có thể của lưới 3x3 hay các cách sắp xếp cụ thể vị trí các ô.
+ Hành động
  - Di chuyển ô trống lên, xuống, trái, hoặc phải để hoán đổi với ô số liền kề.
+ Chi phí
  - Mỗi bước di chuyển có chi phí bằng 1, vì bài toán ưu tiên tìm đường đi ngắn nhất.
+ Giải pháp
  - Dãy các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu, được tạo ra bởi các thuật toán tìm kiếm không có thông tin BFS, DFS, UCS, và IDS.
  
Hình ảnh gif từng thuật toán cùng biểu đồ so sánh hiệu suất

![Nhóm 1](asset/gif/nhom1.gif)

Nhận xét
+ BFS (Breadth-First Search): Hiệu quả trong việc tìm đường đi ngắn nhất nhờ khám phá theo mức độ (level-order). Tuy nhiên, BFS có thể tiêu tốn nhiều bộ nhớ, đặc biệt khi không gian trạng thái lớn, do phải lưu trữ tất cả các trạng thái ở mỗi mức.
+ DFS (Depth-First Search): Tiết kiệm bộ nhớ hơn BFS vì chỉ lưu trữ một đường đi tại một thời điểm. Tuy nhiên, DFS dễ bị đi sâu vào các nhánh không hứa hẹn, dẫn đến hiệu suất kém nếu trạng thái mục tiêu nằm ở độ sâu thấp.
+ UCS (Uniform-Cost Search): Tương tự BFS, UCS đảm bảo tìm được đường đi tối ưu. Tuy nhiên, UCS linh hoạt hơn khi chi phí các bước có thể khác nhau, mặc dù điều này không ảnh hưởng nhiều trong bài toán 8-Puzzle với chi phí đồng nhất.
+ IDS (Iterative Deepening Search): Kết hợp ưu điểm của BFS và DFS, IDS khám phá theo từng mức độ nhưng không tiêu tốn bộ nhớ như BFS. IDS thường hiệu quả hơn trong các bài toán như 8-Puzzle, đặc biệt khi cần cân bằng giữa bộ nhớ và thời gian chạy.
  
Kết luận
+ IDS thường nổi bật nhờ khả năng cân bằng giữa bộ nhớ và hiệu quả tìm kiếm, phù hợp với bài toán 8-Puzzle.
+ DFS có thể kém hiệu quả nhất trong nhóm này, do không đảm bảo tìm được đường đi ngắn nhất và dễ bị kẹt ở các nhánh sâu.

## 2.2  Nhóm thuật toán tìm kiếm có thông tin (Informed Search Algorithms)
Các thành phần chính của bài toán tìm kiếm và giải pháp
+ Trạng thái ban đầu
  - Một lưới 3x3 với 8 số từ 1 đến 8 và một ô trống (0), đại diện cho trạng thái khởi đầu của bài toán ([[2 0 3], [1 4 6], [7 5 8]]).
+ Trạng thái mục tiêu
  - Lưới 3x3 với thứ tự số từ 1 đến 8 và ô trống ở vị trí cuối cùng ([[1 2 3], [4 5 6], [7 8 0]]).
+ Không gian trạng thái
  - Tập hợp tất cả các cấu hình có thể của lưới 3x3 hay các cách sắp xếp cụ thể vị trí các ô.
+ Hành động
  - Di chuyển ô trống lên, xuống, trái, hoặc phải để hoán đổi với ô số liền kề.
+ Chi phí
  - Mỗi bước di chuyển có chi phí bằng 1, vì bài toán ưu tiên tìm đường đi ngắn nhất.
+ Giải pháp
  - Dãy các trạng thái từ trạng thái ban đầu đến trạng thái mục tiêu, được tạo ra bởi các thuật toán tìm kiếm có thông tin GBFS, A*, và IDA*.

Hình ảnh gif từng thuật toán cùng biểu đồ so sánh hiệu suất

![Nhóm 2](asset/gif/nhom2.gif)

Nhận xét
+ GBFS (Greedy Best-First Search): Nhanh và khám phá ít trạng thái nhờ chỉ tập trung vào giá trị heuristic (thường là Manhattan Distance), bỏ qua chi phí đã đi. Với trạng thái ban đầu 203146758, GBFS tìm giải pháp nhanh chóng, nhưng có thể không đảm bảo đường đi ngắn nhất do dễ bị kẹt ở local optima nếu heuristic không đủ chính xác.
+ A*: Đảm bảo tìm đường đi tối ưu nhờ sử dụng hàm đánh giá f(n) = g(n) + h(n), nhưng tiêu tốn nhiều bộ nhớ hơn GBFS và IDA* do duy trì hàng đợi ưu tiên lớn. Với trạng thái này, A* hoạt động ổn định, khám phá số trạng thái trung bình và tìm được đường đi tối ưu, nhưng thời gian chạy lâu hơn GBFS và IDA*.
+ IDA (Iterative Deepening A Star)*: Kết hợp ưu điểm của A* và tìm kiếm theo độ sâu, IDA* tiết kiệm bộ nhớ và chạy rất nhanh nhờ không lưu trữ hàng đợi lớn. Tuy nhiên, với trạng thái 203146758, IDA* khám phá nhiều trạng thái nhất do cơ chế lặp lại (iterative deepening) khiến nó phải quay lại các trạng thái đã khám phá ở các lần lặp trước, đặc biệt nếu heuristic không tối ưu.

Kết luận
+ IDA* nổi bật về tốc độ và tiết kiệm bộ nhớ, nhưng số trạng thái khám phá cao hơn dự kiến cho thấy cần cải thiện heuristic (ví dụ: kết hợp Manhattan Distance với Linear Conflict) để giảm số lần lặp.
+ GBFS phù hợp khi ưu tiên tốc độ, nhưng không đảm bảo đường đi tối ưu, trong khi A* là lựa chọn tốt nhất nếu cần đảm bảo tính tối ưu và sẵn sàng đánh đổi về tài nguyên.

