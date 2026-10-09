# Myanmar Recap Studio — SaaS Frontend Prototype

This folder is an isolated, static frontend prototype for the planned private SaaS website. It is intentionally separate from the current working application files in the repository root.

## Files

- `index.html` — workspace dashboard structure
- `styles.css` — responsive dark cinematic navy/gold design
- `app.js` — demo-only navigation, toast messages, and create-draft modal

## Important limitations

This is **not yet a production SaaS application**. It has no real login, invitations, database, billing, project persistence, or video rendering connection. Sample projects and usage numbers are illustrative only. Do not treat the interface as an access-control or security layer.

Invitation-only access must be enforced by a trusted backend/authentication provider. Admin invitation secrets must never be placed in browser JavaScript. Before inviting real users, implement authentication, server-side role checks, database row-level security, and backend authorization for every protected API.

## Preview locally

Open `index.html` in a browser, or serve this directory using any local static HTTP server.

## Cloudflare Pages preview

For a separate preview deployment from this branch, connect the GitHub repository to Cloudflare Pages and configure:

- Production branch for the main website: keep `main` until ready to launch.
- Build command: `exit 0`
- Build output directory: `saas-frontend`

Cloudflare Pages can also create preview deployments for non-production branches. Verify the branch preview before merging anything into `main`.

## Next implementation stages

1. Choose the final frontend route and deploy a private visual preview.
2. Connect Supabase Auth with invitation-only onboarding.
3. Add database tables and Row Level Security for profiles, projects, plans, and render jobs.
4. Add a trusted backend endpoint for project creation and admin invitations.
5. Move video processing to a background worker and object storage.
6. Add verified billing and server-enforced plan quotas.

Keep the existing root-level app untouched until the replacement is fully tested.