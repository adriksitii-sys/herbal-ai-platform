-- 本草智研 V2：Supabase 数据库初始化
create extension if not exists pgcrypto;

create table if not exists public.submissions (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    submitter text not null default '匿名提交者',
    experiment_id text not null,
    source text not null default '比赛软件模拟',
    inputs jsonb not null,
    outputs jsonb not null default '{}'::jsonb,
    notes text,
    status text not null default 'pending'
        check (status in ('pending','approved','rejected')),
    reviewed_at timestamptz,
    reviewer text,
    rejection_reason text
);
create index if not exists submissions_status_created_idx
    on public.submissions(status, created_at desc);

create table if not exists public.models (
    id uuid primary key default gen_random_uuid(),
    version text not null,
    created_at timestamptz not null default now(),
    artifact_path text not null,
    metrics jsonb not null default '{}'::jsonb,
    sample_count integer not null default 0,
    feature_columns text[] not null,
    target_columns text[] not null,
    feature_bounds jsonb not null,
    is_published boolean not null default false
);
create index if not exists models_published_created_idx
    on public.models(is_published, created_at desc);

alter table public.submissions enable row level security;
alter table public.models enable row level security;
grant usage on schema public to anon, authenticated;
grant select on public.submissions to anon, authenticated;
revoke insert on public.submissions from anon, authenticated;
grant insert (submitter, experiment_id, source, inputs, outputs, notes, status)
    on public.submissions to anon, authenticated;
grant select on public.models to anon, authenticated;

drop policy if exists "public can submit pending records" on public.submissions;
create policy "public can submit pending records"
on public.submissions for insert to anon, authenticated
with check (status = 'pending');

drop policy if exists "public can read approved records" on public.submissions;
create policy "public can read approved records"
on public.submissions for select to anon, authenticated
using (status = 'approved');

drop policy if exists "public can read published model metadata" on public.models;
create policy "public can read published model metadata"
on public.models for select to anon, authenticated
using (is_published = true);

insert into storage.buckets (id, name, public)
values ('model-artifacts', 'model-artifacts', false)
on conflict (id) do nothing;
-- 不为 anon/authenticated 创建模型文件读写策略；仅服务端 service-role 访问私有桶。
