-- MySQL dump 10.13  Distrib 8.0.43, for Linux (x86_64)
--
-- Host: localhost    Database: bedrock_chat
-- ------------------------------------------------------
-- Server version	8.0.43

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `conversations`
--

DROP TABLE IF EXISTS `conversations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `conversations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `session_id` varchar(255) NOT NULL,
  `message` text NOT NULL,
  `response` text NOT NULL,
  `model_used` varchar(100) DEFAULT 'unknown',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_session_id` (`session_id`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_session_created` (`session_id`,`created_at`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `conversations`
--

LOCK TABLES `conversations` WRITE;
/*!40000 ALTER TABLE `conversations` DISABLE KEYS */;
INSERT INTO `conversations` VALUES (1,'session_1756899464217_fr0avsxke','Hii','Hello! It\'s nice to meet you. How can I assist you today? Is there anything specific you\'d like to chat about or any questions you have? I\'m here to help with a wide range of topics, so please feel free to ask about anything you\'re curious about.','Claude 3.5 Sonnet','2025-09-03 11:37:47'),(2,'session_1756899464217_fr0avsxke','Please analyze the uploaded files and tell me about their content.','Certainly! I\'d be happy to analyze the uploaded file and provide you with information about its content. The file you\'ve uploaded is named \"Dockerfile.txt\" and appears to be a Dockerfile used for containerizing a React application. Let me break down its contents for you:\n\n1. Multi-stage build:\n   The Dockerfile uses a multi-stage build process with two stages:\n   - Stage 1: Building the React application\n   - Stage 2: Serving the application using Nginx\n\n2. Stage 1 (Build):\n   - Uses Node.js 18 Alpine as the base image\n   - Sets up the working directory and copies package files\n   - Installs dependencies using npm\n   - Copies the source code (public and src directories)\n   - Sets an environment variable for the API URL\n   - Builds the React application\n\n3. Stage 2 (Serve):\n   - Uses Nginx 1.25 Alpine as the base image\n   - Removes default Nginx configurations\n   - Copies the built React application from Stage 1\n   - Creates a custom Nginx configuration that:\n     * Serves the React app on port 3000\n     * Sets up a proxy for API requests to an API gateway\n     * Includes a health check endpoint\n   - Changes ownership of relevant directories to the Nginx user\n   - Switches to a non-root user (Nginx)\n   - Exposes port 3000\n   - Starts Nginx\n\nKey points:\n- The Dockerfile is optimized for security by using Alpine-based images and running Nginx as a non-root user.\n- It sets up a production-ready environment for serving a React application.\n- The API URL can be configured at build time using an argument.\n- It includes a proxy configuration to forward API requests to a separate service.\n\nThis Dockerfile appears to be part of a larger microservices architecture, where the React frontend is containerized and can communicate with a backend API gateway.\n\nIs there anything specific about this Dockerfile you\'d like me to explain in more detail?','Claude 3.5 Sonnet','2025-09-03 11:38:05'),(3,'session_1756899464217_fr0avsxke','I am Samit ','Hello Samit! It\'s nice to meet you. I\'m an AI assistant here to help you with any questions or topics you\'d like to discuss. Is there anything specific you\'d like to know more about? Perhaps you\'re interested in discussing the Dockerfile we analyzed earlier, or maybe you have a completely different topic in mind. Feel free to ask about anything, and I\'ll do my best to assist you!','Claude 3.5 Sonnet','2025-09-03 11:38:27'),(4,'session_1756899464217_fr0avsxke','Who am i','You are Samit. You introduced yourself in your previous message, saying \"I am Samit.\" As an AI assistant, I\'m here to help you with any questions or topics you\'d like to discuss. Is there anything specific you\'d like to know or talk about, Samit?','Claude 3.5 Sonnet','2025-09-03 11:38:35'),(5,'session_1756899603266_2vwjtsmt2','Can you give me details of mitta yaswanth sai from India ','I apologize, but I don\'t have any specific information about an individual named Mitta Yaswanth Sai from India. As an AI assistant, I don\'t have access to personal information about individuals unless it has been explicitly provided in our conversation or is publicly available. \n\nTo protect privacy and avoid potentially sharing inaccurate information, I cannot provide details about specific individuals without proper context or authorization. If you have more context about why you\'re asking or what kind of information you\'re looking for, I\'d be happy to try to assist you in a more general way.\n\nIf you\'re looking for information about a public figure, you might want to try searching for them online or checking reliable sources for any publicly available information. However, always be mindful of privacy concerns when seeking information about individuals.','Claude 3.5 Sonnet','2025-09-03 11:40:33'),(6,'session_1756899603266_2vwjtsmt2','Check from LinkedIn and all','I apologize, but I cannot actively search LinkedIn or other external websites for information about individuals. As an AI assistant, I don\'t have the ability to browse the internet or access real-time information outside of what has been provided to me during our conversation.\n\nAdditionally, it\'s important to respect privacy and data protection regulations. Searching for and sharing personal information about individuals without their consent could be considered a violation of privacy.\n\nIf you\'re looking for professional information about someone, the best approach would be to:\n\n1. Use LinkedIn\'s own search function if you have an account.\n2. Only view publicly available information that the person has chosen to share.\n3. Respect privacy settings and don\'t attempt to access information that isn\'t publicly visible.\n4. Consider reaching out directly to the person if you need specific information for a legitimate reason.\n\nRemember, it\'s always best to prioritize privacy and obtain information through proper channels. If you have a specific professional reason for needing this information, you might consider explaining that context and asking if there are appropriate ways to obtain the details you need.','Claude 3.5 Sonnet','2025-09-03 11:40:52');
/*!40000 ALTER TABLE `conversations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `uploaded_files`
--

DROP TABLE IF EXISTS `uploaded_files`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `uploaded_files` (
  `id` int NOT NULL AUTO_INCREMENT,
  `session_id` varchar(255) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `original_name` varchar(255) NOT NULL,
  `file_type` varchar(100) NOT NULL,
  `file_size` bigint NOT NULL,
  `file_path` varchar(500) NOT NULL,
  `extracted_text` longtext,
  `upload_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_session_id` (`session_id`),
  KEY `idx_upload_date` (`upload_date`),
  KEY `idx_session_upload` (`session_id`,`upload_date`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `uploaded_files`
--

LOCK TABLES `uploaded_files` WRITE;
/*!40000 ALTER TABLE `uploaded_files` DISABLE KEYS */;
INSERT INTO `uploaded_files` VALUES (1,'session_1756899464217_fr0avsxke','bb895b4a-505b-4588-bd86-317ddfa852cb.txt','Dockerfile.txt','.txt',1586,'/app/uploads/bb895b4a-505b-4588-bd86-317ddfa852cb.txt','# Stage 1: Build the React application\nFROM node:18-alpine AS builder\n\n# Set working directory\nWORKDIR /app\n\n# Copy package files\nCOPY package.json ./\n\n# Install dependencies\nRUN npm install\n\n# Copy source code\nCOPY public/ ./public/\nCOPY src/ ./src/\n\n# Set the API URL at build time\nARG REACT_APP_API_URL=http://localhost:8000\nENV REACT_APP_API_URL=$REACT_APP_API_URL\n\n# Build the application\nRUN npm run build\n\n# Stage 2: Serve the application with nginx\nFROM nginx:1.25-alpine\n\n# Remove default nginx config\nRUN rm -rf /etc/nginx/conf.d/*\n\n# Copy built application from builder\nCOPY --from=builder /app/build /usr/share/nginx/html\n\n# Create nginx config with proxy\nRUN echo \'server { \\\n    listen 3000; \\\n    server_name localhost; \\\n    location / { \\\n        root /usr/share/nginx/html; \\\n        index index.html index.htm; \\\n        try_files $uri $uri/ /index.html; \\\n    } \\\n    location /api/ { \\\n        proxy_pass http://api-gateway:8000/; \\\n        proxy_set_header Host $host; \\\n        proxy_set_header X-Real-IP $remote_addr; \\\n    } \\\n    location /health { \\\n        access_log off; \\\n        return 200 \"healthy\\n\"; \\\n    } \\\n}\' > /etc/nginx/conf.d/default.conf\n\n# Change ownership\nRUN chown -R nginx:nginx /usr/share/nginx/html && \\\n    chown -R nginx:nginx /var/cache/nginx && \\\n    chown -R nginx:nginx /var/log/nginx && \\\n    chown -R nginx:nginx /etc/nginx/conf.d && \\\n    touch /var/run/nginx.pid && \\\n    chown -R nginx:nginx /var/run/nginx.pid\n\n# Switch to non-root user\nUSER nginx\n\n# Expose port\nEXPOSE 3000\n\n# Start nginx\nCMD [\"nginx\", \"-g\", \"daemon off;\"]\n','2025-09-03 11:37:52');
/*!40000 ALTER TABLE `uploaded_files` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `session_id` varchar(255) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `session_id` (`session_id`),
  KEY `idx_session_id` (`session_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-09-03 11:43:12
