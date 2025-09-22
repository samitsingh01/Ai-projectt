# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod main.go ./

# Install dependencies, build, and clean up in minimal layers
RUN apk add --no-cache git ca-certificates && \
    go mod download && \
    go mod tidy && \
    CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o bedrock-service .

# Runtime stage
FROM alpine:3.19

# Install runtime dependencies and create user
RUN apk add --no-cache ca-certificates && \
    addgroup -g 1001 -S appuser && \
    adduser -u 1001 -S appuser -G appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/bedrock-service .

USER appuser
EXPOSE 9000

CMD ["./bedrock-service"]
