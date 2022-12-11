# Tổng quan về Machine Learning

## Objective
* Nắm được hình ảnh tổng quan về lĩnh vực ML và một số ứng dụng thực tế của ML trong cuộc sống
* Demo ứng dụng thực tiễn của ML trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP) giúp ích cho công việc hàng ngày
    * Chi tiết*
        * Chuyển những đoạn hội thoại ghi âm tiếng Nhật thành text
        * Mô hình sử dụng: [Whisper](https://github.com/openai/whisper) (OpenAI)

## I. Machine Learning (ML) là gì ?

* Vài năm trở lại đây, Machine Learning hiện là xu hướng của thể giới với sự ra đời hàng trăm, hàng nghìn các bài báo, các tạp chí nói về lĩnh vực này. 

* Hiện nay đã có nhiều ứng dụng của ML đang len lỏi vào hầu hết các lĩnh vực trong đời sống xã hội như y tế, quản lý an ninh, hóa học, chính trị, điện ảnh… và nó đã chứng minh tiềm năng và hiệu quả thực sự mà ML mang lại cho cuộc sống con người. 

* Theo Wikipedia, Trí tuệ nhân tạo hay AI là một ngành khoa học của khoa học máy tính, là trí thông minh được thể hiện bằng máy móc, trái ngược với trí thông minh tự nhiên được con người thể hiện. Thông thường, thuật ngữ “trí tuệ nhân tạo” thường được sử dụng để mô tả các máy móc (hoặc máy tính) bắt chước các chức năng “nhận thức” mà con người liên kết với tâm trí con người, như “học tập” và “giải quyết vấn đề”. Và ML là một nhóm ngành nhỏ trong AI, chuyên nghiên cứu và xây dựng các kĩ thuật cho phép các hệ thống “học” tự động từ dữ liệu để giải quyết những vấn đề cụ thể như:
    * Làm cho máy tính có những khả năng nhận thức cơ bản của con người như nghe, nhìn, hiểu được ngôn ngữ, giải toán, lập trình, …
    * Hỗ trợ con người trong việc xử lý một khối lượng thông tin khổng lồ mà chúng ta phải đối mặt hàng ngày, hay còn gọi là Big Data

* Machine Learning có liên quan lớn đến thống kê vì cả hai lĩnh vực đều nghiên cứu việc phân tích dữ liệu, nhưng khác với thống kê, Machine Learning tập trung vào sự phức tạp của các giải thuật trong việc thực thi tính toán. Nhiều bài toán suy luận được xếp vào loại bài toán NP-khó, vì thế một phần của Machine Learning là nghiên cứu sự phát triển các giải thuật suy luận xấp xỉ mà có thể xử lý được, hiện nay Machine Learning ứng dụng rất nhiều trong cuộc sống chúng ta như máy truy tìm dữ liệu, chẩn đoán y khoa, phát hiện thẻ tín dụng giả, phân tích thị trường chứng khoán,…

* Túm lại, ML là nhóm ngành nhỏ của Trí tuệ nhân tạo, chúng ta sẽ sử dụng các thuật toán để làm cho máy tính có thể “hiểu” dữ liệu để thực hiện các công việc thay vì thực hiện lập trình một cách tường minh bằng các câu lệnh “if – else”, hay các câu truy vấn thông thường,…

## II. Phân loại thuật toán ML
* Hiện nay, có 2 nhóm chính trong ML đó là: **Học có giám sát (Supervised learning)** và **Học không giám sát (Unsupervised learning)**. Ngoài ra còn có **Semi-Supervised Learning (Học bán giám sát)** và **Reinforcement Learning (Học Củng Cố)**. Điểm khác biệt của các nhóm thuật toán này đó chính là dữ liệu được đưa vào huấn luyện của mô hình, cách thuật toán sử dụng dữ liệu và loại vấn đề mà chúng giải quyết.


<p align="center" width="100%">
    <img width="50%" src="https://blog.luyencode.net/wp-content/uploads/2018/09/phan-loai-machine-learning-768x550.jpg"> 
</p>
<div align="center">
  Phân loại thuật toán ML
</div>


### 1. Học có giám sát (Suppervised learning):
* Học có giám sát là thuật toán học để dự đoán đầu ra mong muốn của một dữ liệu mới (output) dựa vào các điểm dữ liệu (data points) chứa 2 giá trị (input, label) đã biết từ trước. Input được gọi là các đặc trưng của dữ liệu và label chính là nhãn của điểm dữ liệu đó. 
* Để hình dung rõ hơn, chúng ta có thể lấy ví dụ như sau: Xét các thuộc tính x = {diện tích nhà 100m2, 3LDK, nhà koudatsu, cách ga 5p, cách siêu thị 3p ...} và y = {Giá nhà 8,000man}, thì cặp (x,y) được gọi là một điểm dữ liệu, x được gọi là các feature input đầu vào, y chính là nhãn tương ứng với input x. Thông thường dữ liệu của chúng ta là tập hợp của rất nhiều điểm dữ liệu. Bài toán có thể phát biểu một cách cụ thể như sau:
    * Cho một tập hợp các điểm dữ liệu x(1),x(2),x(3),...,x(n) tương ứng với các tập đầu ra là y(1), y(2), y(3),....,y(n).Chúng ta sẽ xây dựng một mô hình học được các dự đoán y từ x dựa vào các tập hợp điểm đã cho trước đó.

* Trong nhóm Học có giám sát, các thuật toán lại chia ra thành 2 nhóm nhỏ hơn đó là: Phân loại (Classification) và Hồi quy (Regresstion). 

<p align="center" width="25%">
    <img width="50%" src="https://lethach.com/wp-content/uploads/2020/07/Capture.png"> 
</p>
<div align="center">
  Phân biệt Hồi quy và Phân loại
</div>

### 2. Học không giám sát (Unsuppervised learning):
* Khác với học có giám sát, dữ liệu của học không giám sát chỉ có các đặc trưng, không có nhãn kèm theo. Unsuppervised Learning được sử dụng để khám phá ra những quy luật ẩn (hidden pattern) trong tập dữ liệu không nhãn. Trong thuật toán này lại được phân thành 2 nhóm nhỏ hơn đó là Gom nhóm (clusstering) và association :
    * Clusstering: Tập trung vào việc gom nhóm các dữ liệu thành k nhóm dựa trên những đặc điểm tương đồng của các điểm dữ liệu. Ví dụ gom nhóm các loại rau củ quả theo hình dạng hoặc màu sắc,…
    * Association: Tìm ra quy luật dựa trên nhiều dữ liệu cho trước, ví dụ: Dựa vào hành vi mua đồ kèm theo của con người ( đàn ông đi chợ mua tả lót thì sẽ mua bia, mua quần sẽ mua kèm cùng với thắt lưng,…) từ đó xây dựng một hệ thống gợi ý cho sản phẩm tiếp theo, hoặc bố trí sản phẩm cho phù hợp

### 3. Semi-Supervised Learning (Học bán giám sát):
* Học bán giám sát có dữ liệu để học không đầy đủ nhãn. Nghĩa là chỉ một phần của chúng được gán nhãn. Những bài toán thuộc nhóm này nằm giữa hai nhóm được nêu bên trên. Thực tế cho thấy rất nhiều các bài toán Machine Learning thuộc vào nhóm này vì việc thu thập dữ liệu có nhãn tốn rất nhiều thời gian và có chi phí cao. Rất nhiều loại dữ liệu thậm chí cần phải có chuyên gia mới gán nhãn được (ảnh y học chẳng hạn). Ngược lại, dữ liệu chưa có nhãn có thể được thu thập với chi phí thấp từ internet.

### 4. Reinforcement Learning (Học Củng Cố):
* Reinforcement learning là các bài toán giúp cho một hệ thống tự động xác định hành vi dựa trên hoàn cảnh để đạt được lợi ích cao nhất (maximizing the performance). Hiện tại, Reinforcement learning chủ yếu được áp dụng vào Lý Thuyết Trò Chơi (Game Theory), các thuật toán cần xác định nước đi tiếp theo để đạt được điểm số cao nhất.

* Một số ứng dụng nổi tiếng hiện nay áp dụng học củng cố phải kể đến là AlphaGo. AlphaGo có thể tự chơi với nó hàng triệu để tìm ra các nước đi tối ưu hơn dựa trên các nước đi của con người.

* [Ví dụ](https://machinelearningcoban.com/2016/12/27/categories/#supervised-learning-hoc-co-giam-sat)

## III. Ứng dụng của ML trong cuộc sống con người
* Hiện nay ML đã có mặt hầu hết trong cuộc sống của chúng ta, len lỏi hầu hết ở các lĩnh vực trong đời sống xã hội, tiêu biểu là các lĩnh vực sau:

### 1. Xử lý ảnh: Trích xuất các thông tin có trong các bức ảnh, ví dụ:
* Gắn thẻ hình ảnh: Như các bạn đã biết, hiện nay facebook có tính năng auto tag Tên của bạn bè khi mình đăng ảnh. Về cơ bản, thuật toán đã trích xuất những đặc trưng của các bức ảnh mà bạn đã đăng trước đó, sau đó tiến hành so khớp để gán nhãn tự động.
* Nhận diện chữ viết tay: Cũng giống như auto tag, nhận diện chữ viết tay cũng trích xuất nhưng đặc trưng của hình ảnh chữ viết tay trước đó, sau đó tiến hành xử lý, nhận dạng ký tự khi có một bức ảnh có chữ viết tay đưa vào
* Xe tự hành: Thuật toán ML giúp phát hiện mép đường, vật cản, biển báo bằng cách xử lý các khung hình được ghi lại bởi cammera

### 2. Phân tích văn bản: Trích xuất thông tin từ văn bản
* Lọc spam: mail spam hay không spam,..
* Khai phá thông tin: Trích xuất các từ khóa, tên, địa chỉ,…
* Phân tích ngữ nghĩa: tiêu cực hay tích cực,…

### 3. Khai phá dữ liệu: là quá trình khám phá ra các thông tin có giá trị hoặc đưa ra các dự đoán từ dữ liệu. Nghĩa là tìm kiếm các thông tìn hữu ý từ một tập dữ liệu rất lớn
* Phát hiện bất thường trong chứng khoán, giao dịch ngân hàng
* Phát hiện các quy luật: tìm quy luật về việc mua hàng hóa, từ đó xây dựng hệ thống khuyên nghị phù hợp,..
* Gom nhóm: Gom nhóm phân khúc mua hàng dựa vào độ tuổi, kinh tế để có chiến lược makerting phù hợp
* Dự đoán: giá cổ phiếu, giá nhà

### 4. Robot:
* Các sản phẩm về máy hút bụi thông minh, …

### IV. Demo
* [Link ví dụ]()