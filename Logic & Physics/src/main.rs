use anyhow::{Context, Result};
use clap::Parser;
use image::{ImageBuffer, Rgba, RgbaImage};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    #[arg(long, default_value_t = 1000000)]
    pub balls: u64,

    #[arg(long, default_value = "galton_board.png")]
    pub out: String,

    #[arg(long)]
    pub workers: Option<usize>,

    #[arg(long, default_value_t = 1920)]
    pub width: u32,

    #[arg(long, default_value_t = 1080)]
    pub height: u32,

    #[arg(long, default_value_t = 100)]
    pub levels: i32,

    #[arg(long, default_value_t = 2.0)]
    pub peg_radius: f64,

    #[arg(long, default_value_t = 1.5)]
    pub ball_radius: f64,

    #[arg(long, default_value_t = 18.0)]
    pub peg_spacing_x: f64,

    #[arg(long, default_value_t = 15.0)]
    pub peg_spacing_y: f64,

    #[arg(long, default_value_t = 981.0)]
    pub gravity: f64,

    #[arg(long, default_value_t = 0.6)]
    pub restitution: f64,

    #[arg(long, default_value_t = 10.0)]
    pub fuzz: f64,

    #[arg(long, default_value_t = 0.005)]
    pub dt: f64,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            anyhow::bail!("dimensions must be positive");
        }
        if let Some(w) = self.workers {
            if w == 0 {
                anyhow::bail!("workers must be positive");
            }
        }
        if self.balls == 0 {
            anyhow::bail!("balls must be positive");
        }
        if self.levels <= 0 {
            anyhow::bail!("levels must be positive");
        }
        if self.peg_radius <= 0.0 || self.ball_radius <= 0.0 {
            anyhow::bail!("radii must be positive");
        }
        if self.peg_spacing_x <= 0.0 || self.peg_spacing_y <= 0.0 {
            anyhow::bail!("spacing must be positive");
        }
        if self.gravity <= 0.0 {
            anyhow::bail!("gravity must be positive");
        }
        if self.dt <= 0.0 {
            anyhow::bail!("dt must be positive");
        }
        if self.out.is_empty() {
            anyhow::bail!("path must not be empty");
        }
        Ok(())
    }
}

pub struct Simulation {
    cfg: Config,
}

impl Simulation {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn simulate(&self) -> Vec<u64> {
        let dist = (0..self.cfg.width)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>();

        if let Some(w) = self.cfg.workers {
             let _ = rayon::ThreadPoolBuilder::new().num_threads(w).build_global();
        }

        (0..self.cfg.balls).into_par_iter().for_each(|_| {
            let mut rng = Pcg64::from_entropy();
            let bin = self.simulate_ball(&mut rng);
            if bin < self.cfg.width as usize {
                dist[bin].fetch_add(1, Ordering::Relaxed);
            }
        });

        dist.into_iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    fn simulate_ball(&self, rng: &mut Pcg64) -> usize {
        let mut x = self.cfg.width as f64 / 2.0;
        x += (rng.gen::<f64>() - 0.5) * 0.1;
        let mut y = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;

        let max_y = self.cfg.levels as f64 * self.cfg.peg_spacing_y;
        let min_dist = self.cfg.peg_radius + self.cfg.ball_radius;
        let min_dist_sq = min_dist * min_dist;
        let half_dt_sq_gravity = 0.5 * self.cfg.gravity * self.cfg.dt * self.cfg.dt;

        while y < max_y {
            x += vx * self.cfg.dt;
            y += vy * self.cfg.dt + half_dt_sq_gravity;
            vy += self.cfg.gravity * self.cfg.dt;

            if x < 0.0 {
                x = 0.0;
                vx = -vx * self.cfg.restitution;
            } else if x > (self.cfg.width - 1) as f64 {
                x = (self.cfg.width - 1) as f64;
                vx = -vx * self.cfg.restitution;
            }

            let row = (y / self.cfg.peg_spacing_y).round() as i32;
            if row >= 0 && row < self.cfg.levels {
                let peg_y = row as f64 * self.cfg.peg_spacing_y;
                let offset = if row & 1 != 0 {
                    self.cfg.peg_spacing_x / 2.0
                } else {
                    0.0
                };

                let col = ((x - offset) / self.cfg.peg_spacing_x).round() as i32;
                let peg_x = col as f64 * self.cfg.peg_spacing_x + offset;

                let dx = x - peg_x;
                let dy = y - peg_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist_sq && dist_sq > 0.0001 {
                    let d = dist_sq.sqrt();
                    let nx = dx / d;
                    let ny = dy / d;

                    let overlap = min_dist - d;
                    x += nx * overlap;
                    y += ny * overlap;

                    let dot = vx * nx + vy * ny;
                    if dot < 0.0 {
                        vx = (vx - 2.0 * dot * nx) * self.cfg.restitution;
                        vy = (vy - 2.0 * dot * ny) * self.cfg.restitution;
                        vx += (rng.gen::<f64>() - 0.5) * self.cfg.fuzz;
                        vy += (rng.gen::<f64>() - 0.5) * self.cfg.fuzz;
                    }
                }
            }
        }

        let bin = x.round() as i32;
        if bin < 0 {
            0
        } else if bin >= self.cfg.width as i32 {
            (self.cfg.width - 1) as usize
        } else {
            bin as usize
        }
    }
}

pub struct Renderer {
    cfg: Config,
}

impl Renderer {
    pub fn new(cfg: Config) -> Self {
        Self { cfg }
    }

    pub fn render(&self, dist: &[u64]) -> RgbaImage {
        let mut img = ImageBuffer::from_pixel(self.cfg.width, self.cfg.height, Rgba([10, 10, 15, 255]));

        let max_count = dist.iter().max().copied().unwrap_or(0);
        if max_count == 0 {
            return img;
        }

        let color_map = self.generate_color_map();
        let inv_max_count = 1.0 / max_count as f64;
        let f_height = self.cfg.height as f64;

        let mut bar_heights = vec![0; self.cfg.width as usize];
        for x in 0..self.cfg.width as usize {
            if x < dist.len() {
                let mut bh = (dist[x] as f64 * inv_max_count * f_height) as u32;
                if bh > self.cfg.height {
                    bh = self.cfg.height;
                }
                bar_heights[x] = bh;
            }
        }

        for py in 0..self.cfg.height {
            let y = self.cfg.height - 1 - py;
            let c = color_map[y as usize];
            for x in 0..self.cfg.width {
                if y < bar_heights[x as usize] {
                    img.put_pixel(x, py, c);
                }
            }
        }
        img
    }

    fn lerp(&self, a: f64, b: f64, t: f64) -> u8 {
        (a + (b - a) * t) as u8
    }

    fn generate_color_map(&self) -> Vec<Rgba<u8>> {
        let mut cmap = Vec::with_capacity(self.cfg.height as usize);
        let inv_height = 1.0 / self.cfg.height as f64;
        for y in 0..self.cfg.height {
            let t = y as f64 * inv_height;
            let (red, green, blue);
            if t < 0.5 {
                let t2 = t * 2.0;
                red = self.lerp(10.0, 220.0, t2);
                green = self.lerp(10.0, 20.0, t2);
                blue = self.lerp(40.0, 60.0, t2);
            } else {
                let t2 = t * 2.0 - 1.0;
                red = self.lerp(220.0, 255.0, t2);
                green = self.lerp(20.0, 215.0, t2);
                blue = self.lerp(60.0, 0.0, t2);
            }
            cmap.push(Rgba([red, green, blue, 255]));
        }
        cmap
    }
}

fn main() -> Result<()> {
    let cfg = Config::parse();
    cfg.validate().context("Configuration validation failed")?;

    let start_time = Instant::now();
    let sim = Simulation::new(cfg.clone());
    let distribution = sim.simulate();

    let renderer = Renderer::new(cfg.clone());
    let img = renderer.render(&distribution);

    let out_path = Path::new(&cfg.out);
    img.save(out_path).context("Failed to save image")?;

    println!(
        "Completed in {:?}. Saved to {}",
        start_time.elapsed(),
        cfg.out
    );

    Ok(())
}
