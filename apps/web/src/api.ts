export type Camera = {id:string; name:string; location:string; timezone:string; clock_offset_ms:number}
export type Video = {id:string; camera_id:string; camera_name:string; location:string; filename:string; status:string; duration_ms:number; frame_count:number; created_at:string; error?:string; sha256:string; source_started_at_utc?:string}
export type Job = {id:string; video_id:string; stage:string; status:string; processed:number; total:number; elapsed_ms:number; error?:string; created_at:string; updated_at:string}
export type Match = {id:string; video_id:string; camera_id:string; camera_name:string; location:string; filename:string; start_ms:number; end_ms:number; pts_ms:number; similarity_raw:number; review_status:string; review_note:string; source_utc?:string; thumbnail_url:string; clip_url:string; sha256:string}
export type SearchResult = {search_id:string|null; kind?:string; query?:string; reference_id?:string; answer_status:string; summary:string; limitations:string; matches:Match[]}
export type Overview = {camera_count:number; video_count:number; ready_count:number; frame_count:number; recent_videos:Video[]; jobs:Job[]}

export class Api {
  token:string
  constructor(token:string){this.token=token}
  async request<T>(path:string, init:RequestInit={}):Promise<T>{
    const headers=new Headers(init.headers); headers.set('Authorization',`Bearer ${this.token}`)
    if(init.body && !(init.body instanceof FormData))headers.set('Content-Type','application/json')
    const response=await fetch('/api/v1'+path,{...init,headers})
    if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();message=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}catch{}throw new Error(message)}
    return response.json()
  }
  get<T>(path:string){return this.request<T>(path)}
  post<T>(path:string,body:unknown){return this.request<T>(path,{method:'POST',body:JSON.stringify(body)})}
  upload<T>(path:string,body:FormData){return this.request<T>(path,{method:'POST',body})}
  async blob(path:string){const response=await fetch('/api/v1'+path,{headers:{Authorization:`Bearer ${this.token}`}});if(!response.ok)throw new Error('Media could not be opened');return response.blob()}
}
