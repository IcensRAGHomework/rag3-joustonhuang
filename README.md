[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SQDH8b2K)
# Chromadb 作業題目

## 作業內容

請使用 **chromadb** 套件完成以下作業，將 `COA_OpenData.csv` 檔案寫入 chroma.sqlite3(需要上傳該檔案)，並在 **`student_assignment.py`** 中實作以下方法：`generate_hw01-03(question)`
#### 創建 `.env` 文件（供學生使用）

在此作業中，您將需要在本地開發環境中設置必要的參數來支持程式運行。為了簡化環境變數的管理，請在項目根目錄下創建一個名為 `.env` 的文件，並在其中定義環境變數。

這個 `.env` 文件的主要用途是為了讓您能夠在自己的電腦上進行實作，並為`model_configurations.py`文件提供所需的參數。參加作業時，我們會提供具體的參數值供您填寫。以下是 .env 文件的範例格式：


```makefile
AZURE_OPENAI_EMBEDDING_ENDPOINT=your_endpoint_here
AZURE_OPENAI_EMBEDDING_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_EMBEDDING=your_deployment_embedding_here
AZURE_OPENAI_VERSION=your_api_version_here
AZURE_OPENAI_DEPLOYMENT_EMBEDDING_MODEL_NAME="text-embedding-ada-002"
AZURE_OPENAI_TYPE=your_openai_type_here
```
#### 注意事項

- **請勿將 `.env` 文件上傳到任何版本控制系統（例如 GitHub）**，以避免洩漏敏感資訊。
- `.env` 文件僅供您在本地環境中使用，不需要提交作業時包含在內。


---

### 作業1：初始化資料庫並添加 Metadata

1. **題目描述**：需要先將數據存入資料庫（ChromaDB）。這些數據包括店家的描述文字、基本資訊（如名稱、類型、地址等），這些內容將作為查詢和過濾的基礎。
2. **步驟說明**
   - 1. 使用指定的 CSV 檔案
      請使用指定的 CSV 檔案，檔案名稱為 `COA_OpenData.csv`。該檔案已包含所有需要的欄位資料。
   - 2. 建立 Collection
         在將資料寫入資料庫之前，需要先建立或獲取一個 Collection。請確保使用以下參數來建立 Collection：
          - **name**: `"TRAVEL"`，這是此 Collection 的名稱，用於標識此數據集的用途。
          - **metadata**: `{"hnsw:space": "cosine"}`，這是設定查詢相似度計算的參數，`cosine` 表示使用餘弦相似度來進行距離計算。
   - 3. Metadata 的內容
      在初始化資料庫時，需從 CSV 檔案中提取每條記錄的相關欄位，並將其作為 Metadata 存入 ChromaDB。Metadata 包含以下資訊：
         - **file_name**：資料來源檔案名稱（`COA_OpenData.csv`）。
         - **name**：店家名稱。
         - **type**：店家類型，例如 "美食"、"旅遊"。
         - **address**：店家地址。
         - **tel**：店家聯絡電話。
         - **city**：店家所在城市。
         - **town**：店家所在城鎮。
         - **date**：資料創建日期，需從 `CreateDate` 欄位轉換為時間戳格式（秒）。
   - 4. 文件數據（`documents`）
     將 CSV 檔案中的 `HostWords` 欄位內容提取作為文本數據存入 ChromaDB。這些數據是查詢時進行相似度計算的核心。
3. **方法**：實作 `generate_hw01()` 方法，回傳一個collection物件。
4. **輸出格式**：回傳一個 collection物件`chromadb.api.models.Collection.Collection` (需要先上傳chroma.sqlite3該檔案)
---

### 作業2

1. **題目描述**：`根據文件內容查詢特定類型的店家，並過濾出相似度分數在 **0.80** 分以上的結果(請用list格式呈現，順序依照相似度分數遞減排序)`
2. **範例**：`我想要找有關茶餐點的店家`
3. **方法**：實作 `generate_hw02(question, city, store_type, start_date, end_date)` 方法，完成如下功能：
   - 接受用戶輸入的問題及篩選條件，從資料庫中查詢符合條件的店家。
   - 驗證答案時，輸入的篩選參數不一定都會使用。
   - 僅返回符合相似度分數大於或等於 **0.80** 的店家名稱。
   - 設定查詢結果數量為 `10`（`n_results=10`）。
4. **參數**:
   - `question` (str)：用戶的查詢問題，例如 `"我想要找有關茶餐點的店家"`。
   - `city` (list)：需要篩選的城市列表，例如 `["宜蘭縣", "新北市"]`。
   - `store_type` (list)：需要篩選的店家類型列表，例如 `["美食"]`。
   - `start_date` (datetime.datetime)：篩選的開始日期，例如 `datetime.datetime(2024, 4, 1)`。
   - `end_date` (datetime.datetime)：篩選的結束日期，例如 `datetime.datetime(2024, 5, 1)`。
   
5. **輸出格式**：
   - 格式如下：
     ```python
     ['茶之鄉', '山舍茶園', '快樂農家米食坊', '海景咖啡簡餐', '田園香美食坊', '玉露茶驛站', '一佳村養生餐廳', '北海驛站石農肉粽']
     ```

---

### 作業3

1. **題目描述**：學生需要完成以下兩件事情：
   1. 更新資料庫中指定店家的資訊。
   2. 根據查詢條件，列出的店家名稱要根據新的參數來決定顯示的名稱，並過濾出相似度分數在 **0.80** 分以上的結果(請用list格式呈現，順序依照相似度分數遞減排序)`
2. **方法**：
   - 實作 `generate_hw03(question, store_name, new_store_name, city, store_type)` 方法，完成如下功能：  
     1. 找到指定的店家，並在Metadata新增新的參數，名稱為 `new_store_name`。  
     2. 透過問題取得的店家名稱，如果該店家的 Metadata 有 `new_store_name` 參數，請用該參數來顯示新的店家名稱。  
     3. 設定查詢結果數量為 `10`（`n_results=10`）。
3. **參數**:
   - `question` (str)：用戶的查詢問題，例如 `"我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵"`。
   - `store_name` (str)：需要搜尋的店家名稱， 例如 `"耄饕客棧"`。
   - `new_store_name` (str)：需要增加的參數名稱， 例如 `"田媽媽（耄饕客棧）"`。
   - `city` (list)：需要篩選的城市列表，例如 `["南投縣"]`。
   - `store_type` (list)：需要篩選的店家類型列表，例如 `["美食"]`。
4. **輸出格式**：
   - 格式如下
     ```python
     ['田媽媽社區餐廳', '圓夢工坊', '桑園工坊', '田媽媽（耄饕客棧）', '仁上風味坊', '田媽媽美食館']
     ```

---

### 注意事項
- 必須使用 **Chromadb** 套件完成方法實作。
- 確保輸出的格式與範例一致。

### 參考來源
- [Chromadb](https://docs.trychroma.com/guides)

