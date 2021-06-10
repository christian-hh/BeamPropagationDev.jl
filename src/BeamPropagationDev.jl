module BeamPropagationDev

using StaticArrays, Unitful, LoopVectorization, StatsBase, StructArrays

macro params(fields_tuple)
    fields = fields_tuple.args
    esc(
        quote
            NamedTuple{Tuple($fields)}(($fields_tuple))
        end)
end
export @params

macro with_unit(arg1, arg2)
    arg2 = @eval @u_str $arg2
    return convert(Float64, upreferred(eval(arg1) .* arg2).val)
end
export @with_unit

function dtstep!(rs, vs, as, dt)
    @avx for i in eachindex(rs, vs, as)
        rs[i] += vs[i] * dt + as[i] * (0.5 * dt^2)
        vs[i] += as[i] * dt
    end
    return nothing
end

function dtstep!(particles, dt)
    @avx for i in 1:size(particles, 1)
        r = particles.r[i]
        v = particles.v[i]
        a = particles.a[i]
        particles.r[i] += v * dt + a * (0.5 * dt^2)
        particles.v[i] += a * dt
    end
    return nothing
end

function update_dead!(rs, vs, dead, is_dead)
    @inbounds for i in 1:size(rs,1)
        dead[i] = is_dead(rs[i], vs[i])
    end
    return nothing
end

function save!(save, idxs, rs, vs, as, s)
    @inbounds for i in 1:size(rs,1)
        save(idxs[i], rs[i], vs[i], as[i], s)
    end
    return nothing
end

function setup(idxs, r, v, a, p)
    dead_copy   = zeros(Bool, length(idxs))
    rs_copy     = deepcopy(rs[idxs])
    vs_copy     = deepcopy(vs[idxs])
    as_copy     = deepcopy(as[idxs])
    p_copy      = deepcopy(p)
    return rs_copy, vs_copy, as_copy, dead_copy, p_copy, idxs
end

function initialize_dists(n, r, v, a)
    rx = rand(r[1], n); ry = rand(r[2], n); rz = rand(r[3], n)
    vx = rand(v[1], n); vy = rand(v[2], n); vz = rand(v[3], n)
    ax = rand(a[1], n); ay = rand(a[2], n); az = rand(a[3], n)

    rs = SVector.(rx, ry, rz)
    vs = SVector.(vx, vy, vz)
    as = SVector.(ax, ay, az)

    dead = zeros(Bool, n)
    idxs = collect(1:n)

    return rs, vs, as, dead, idxs
end
export initialize_dists

function copy_save_data(s, chunk_idxs)
    s_copy = NamedTuple()
    for key in keys(s)
        array_to_save = deepcopy(s[key][chunk_idxs])
        s_copy = (; s_copy..., key => array_to_save)
    end
    return s_copy
end

function write_data(s, s_copy, chunk_idxs)
    for key in keys(s)
        s[key][chunk_idxs] .= s_copy[key]
    end
    return nothing
end

function delete_data(rs, vs, as, idxs, dead)
    deleteat!(rs, dead)
    deleteat!(vs, dead)
    deleteat!(as, dead)
    deleteat!(idxs, dead)
    deleteat!(dead, dead)
    return nothing
end

function propagate!(n, dt, r, v, a, f, save, discard, save_every, delete_every, max_steps, p, s_initial, s_final)

    chunk_size = round(Int64, n / Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()

        start_idx   = (i-1)*chunk_size+1
        end_idx     = min(i*chunk_size, n)
        chunk_idxs  = start_idx:end_idx
        actual_chunk_size = length(chunk_idxs)

        rs, vs, as, dead, idxs = initialize_dists(actual_chunk_size, r, v, a)
        #states = s_initial.states # overwrite the initialization of states, which put everything in state "1".

        s_initial_copy = copy_save_data(s_initial, chunk_idxs)
        save!(save, idxs, rs, vs, as, s_initial_copy)
        write_data(s_initial, s_initial_copy, chunk_idxs)

        # Do one round of discards before propagation starts
        update_dead!(rs, vs, dead, discard)
        delete_data(rs, vs, as, idxs, dead)

        # Copy parameters and data arrays to be saved to avoid thread race conditions
        s_copy = copy_save_data(s_final, chunk_idxs)
        p_copy = deepcopy(p)

        #propagate_nosave(dt, rs, vs, as, states, dead, idxs, f, discard, delete_every, max_steps, p_copy, s_copy)
        propagate_withsave(dt, rs, vs, as, dead, idxs, f, discard, save_every, delete_every, max_steps, p_copy, s_copy, save)
        write_data(s_final, s_copy, chunk_idxs)

    end

    return s_initial, s_final
end
export propagate!

function propagate_withsave(dt, rs, vs, as, dead, idxs, f, discard, save_every, delete_every, max_steps, p, s, save)
    step = 0
    while (step <= max_steps)
        if (step % save_every == 0) | (step == max_steps)
            save!(save, idxs, rs, vs, as, s)
        end

        if step % delete_every == 0
            update_dead!(rs, vs, dead, discard)
            delete_data(rs, vs, as, idxs, dead)
        end

        # This should ideally be abstracted out into an `update!` function, defined above. Replacing the code below with `update!(f, idxs, rs, vs, as, states, dt, p, s)` incurs allocations, which can be narrowed down to the arguments `s` and `p`, which are both `NamedTuple`s
        @inbounds for i in 1:size(rs,1)
            v′, a′ = f(idxs[i], rs[i], vs[i], as[i], dt, p, s)
            vs[i] = v′
            as[i] = a′
        end

        dtstep!(rs, vs, as, dt)
        step += 1
    end
    return nothing
end
export propagate_withsave

end
