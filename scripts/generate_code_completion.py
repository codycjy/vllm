#!/usr/bin/env python3
"""Generate synthetic code completion data for prefix cache benchmarking.

Simulates multiple users editing files in the same repository:
- Shared prefix: repo context (imports, class definitions) per "file"
- Unique suffix: different cursor positions / completion targets

Output: vLLM custom JSONL format
"""

import argparse
import json
import random
from pathlib import Path

# Simulated "file contexts" acting as shared prefixes
FILE_CONTEXTS = [
    # Python web server
    '''\
import os
import sys
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///app.db"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="User Management API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[\\w.-]+@[\\w.-]+\\.\\w+$")


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

''',
    # Python data pipeline
    '''\
import os
import json
import csv
import logging
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    batch_size: int = 1000
    max_workers: int = 4
    date_format: str = "%Y-%m-%d"
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    def __init__(self, schema: Dict[str, type]):
        self.schema = schema
        self.error_count = 0
        self.total_count = 0

    def validate_record(self, record: Dict[str, Any]) -> bool:
        self.total_count += 1
        for key, expected_type in self.schema.items():
            if key not in record:
                self.error_count += 1
                return False
            if not isinstance(record[key], expected_type):
                try:
                    expected_type(record[key])
                except (ValueError, TypeError):
                    self.error_count += 1
                    return False
        return True

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.total_count, 1)


class TransformPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.transforms: list[Callable] = []
        self.stats = {"processed": 0, "filtered": 0, "errors": 0}

    def add_transform(self, fn: Callable) -> "TransformPipeline":
        self.transforms.append(fn)
        return self

''',
    # JavaScript React component
    '''\
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { z } from 'zod';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import {
  Card, CardContent, CardHeader, CardTitle,
  Button, Input, Label, Select, SelectContent,
  SelectItem, SelectTrigger, SelectValue,
  Table, TableBody, TableCell, TableHead,
  TableHeader, TableRow,
  Dialog, DialogContent, DialogHeader, DialogTitle,
  Badge, Skeleton, Alert, AlertDescription,
  Tabs, TabsContent, TabsList, TabsTrigger,
} from '@/components/ui';
import { apiClient } from '@/lib/api';
import { formatDate, formatCurrency, cn } from '@/lib/utils';
import { useAuth } from '@/hooks/useAuth';
import { useDebounce } from '@/hooks/useDebounce';
import { toast } from 'sonner';

const orderSchema = z.object({
  customerId: z.string().min(1, 'Customer is required'),
  items: z.array(z.object({
    productId: z.string(),
    quantity: z.number().min(1).max(999),
    price: z.number().positive(),
  })).min(1, 'At least one item required'),
  shippingAddress: z.object({
    street: z.string().min(1),
    city: z.string().min(1),
    state: z.string().length(2),
    zipCode: z.string().regex(/^\\d{5}(-\\d{4})?$/),
  }),
  notes: z.string().optional(),
});

const fetchOrders = async ({ page, limit, search, status }) => {
  const params = new URLSearchParams({ page, limit, ...(search && { search }), ...(status && { status }) });
  const { data } = await apiClient.get(`/api/orders?${params}`);
  return data;
};

''',
    # Go HTTP handler
    '''\
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"

	"myapp/internal/auth"
	"myapp/internal/database"
	"myapp/internal/models"
	"myapp/internal/validator"
)

type Handler struct {
	db     database.Store
	auth   *auth.Service
	logger *slog.Logger
	config *Config
}

type Config struct {
	JWTSecret     string
	TokenExpiry   time.Duration
	MaxPageSize   int
	DefaultPage   int
	RateLimit     int
	AllowedOrigins []string
}

func NewHandler(db database.Store, authSvc *auth.Service, logger *slog.Logger, cfg *Config) *Handler {
	return &Handler{db: db, auth: authSvc, logger: logger, config: cfg}
}

func (h *Handler) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(30 * time.Second))

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/auth/login", h.Login)
		r.Post("/auth/register", h.Register)
		r.Post("/auth/refresh", h.RefreshToken)

		r.Group(func(r chi.Router) {
			r.Use(h.AuthMiddleware)
			r.Get("/users/me", h.GetCurrentUser)
			r.Put("/users/me", h.UpdateCurrentUser)
			r.Get("/users/{userID}", h.GetUser)
			r.Get("/products", h.ListProducts)
			r.Post("/products", h.CreateProduct)
			r.Get("/orders", h.ListOrders)
			r.Post("/orders", h.CreateOrder)
		})
	})

	return r
}

''',
    # Rust service
    '''\
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use axum::{
    extract::{Path, Query, State, Json},
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post, put, delete},
    Router,
};
use sqlx::{PgPool, FromRow};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error, instrument};
use validator::Validate;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".into()),
            AppError::Database(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };
        (status, Json(serde_json::json!({"error": message}))).into_response()
    }
}

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    pub config: AppConfig,
}

#[derive(Clone, Deserialize)]
pub struct AppConfig {
    pub database_url: String,
    pub jwt_secret: String,
    pub port: u16,
    pub max_connections: u32,
}

#[derive(Serialize, Deserialize, FromRow)]
pub struct Task {
    pub id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub priority: i32,
    pub assignee_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

''',
]

# Completion tasks appended after each file context
COMPLETION_TASKS = [
    "# TODO: Implement the endpoint to list all users with pagination\n",
    "# TODO: Add error handling for database connection failures\n",
    "# TODO: Implement the search functionality with filters\n",
    "# TODO: Add caching layer for frequently accessed data\n",
    "# TODO: Implement batch processing for bulk operations\n",
    "# TODO: Add input validation and sanitization\n",
    "# TODO: Implement rate limiting middleware\n",
    "# TODO: Add logging and metrics collection\n",
    "# TODO: Implement the delete endpoint with soft delete\n",
    "# TODO: Add unit tests for the service layer\n",
    "# TODO: Implement webhook notification system\n",
    "# TODO: Add authentication middleware\n",
    "# TODO: Implement data export functionality\n",
    "# TODO: Add health check endpoint\n",
    "# TODO: Implement retry logic for external API calls\n",
]


def generate_dataset(args) -> list[dict]:
    rng = random.Random(args.seed)
    contexts = FILE_CONTEXTS[:args.num_files]
    records = []

    for _ in range(args.num_requests):
        file_idx = rng.randrange(len(contexts))
        task = rng.choice(COMPLETION_TASKS)
        prompt = contexts[file_idx] + task
        records.append({
            "prompt": prompt,
            "output_tokens": args.output_tokens,
            "prefix_group": f"file_{file_idx}",
        })

    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic code completion data")
    parser.add_argument("--num-files", type=int, default=5,
                        help="Number of shared file contexts (max 5)")
    parser.add_argument("--num-requests", type=int, default=500,
                        help="Total completion requests")
    parser.add_argument("--output-tokens", type=int, default=256,
                        help="Expected output tokens per completion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    args.num_files = min(args.num_files, len(FILE_CONTEXTS))

    records = generate_dataset(args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    groups = {}
    for r in records:
        groups[r["prefix_group"]] = groups.get(r["prefix_group"], 0) + 1
    print(f"Generated {len(records)} code completion requests → {args.output}")
    print(f"  Files: {len(groups)}")
    for g, c in sorted(groups.items()):
        print(f"    {g}: {c} requests")


if __name__ == "__main__":
    main()
